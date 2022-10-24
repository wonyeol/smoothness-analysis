(** pyppai: basic abstract interpreter for python probabilistic programs
 **
 ** GNU General Public License
 **
 ** Authors:
 **  Wonyeol Lee, KAIST
 **  Xavier Rival, INRIA Paris
 **  Hongseok Yang, KAIST
 **  Hangyeol Yu, KAIST
 **
 ** Copyright (c) 2019 KAIST and INRIA Paris
 **
 ** ai_diff.ml: entry point for continuity/differentiability analysis *)
open Analysis_sig
open Ir_sig
open Ir_util
open Lib

open Ddom_sig
open Ddom_num
open Ddom_diff

open Diff_util


let sep_1 = String.make 78 '=' ^ "\n"
let sep_2 = String.make 78 '-' ^ "\n"

(** ASSUMPTIONS on model and guide passed to our analyser:
 *
 * 1. Support of model and guide:
 *    The support of a guide is a subset of a model, i.e., any sampled value from a guide
 *    has a non-zero probability density in a model.
 *    - If this assumption does not hold, ELBO may not be well-defined.
 *    - This assumption also ensures that any sampled value of a latent variable
 *      in a guide is in the support of the corresponding latent variable in a model.
 *      (It is obvious that the sampled value is in the support of its latent variable
 *      in a guide.) So, it is sound to consider differentiability over the support of
 *      each distribution, and not over a superset of the support.
 *
 * 2. Observed values:
 *    For each statement `pyro.sample(_, d, obs=v)`, (i) the observed value v is a constant
 *    and (ii) the probability density of the observing distribution d at v is non-zero.
 *    - (i) is a common assumption. If not satisfied (e.g., v depends on latent variables
 *      or parameters to be trained), then unexpected behavior may arise.
 *    - (ii) guarantees that v is in the support of d. So, it is again sound to consider
 *      differentiability over the support of each distribution.
 *    - NOTE: guaranteed by our analyser, not by assumptions, is that parameters of
 *            each distribution for `sample` or `observe` are in their supposed domain.
 *
 * 3. Uniform distributions:
 *    All parameters of all Uniform distributions are constant, i.e., do not depend on
 *    any of the parameters to be trained, nor on any of the other latent variables.
 *    - If this assumption does not hold, selective reparameterisation can be biased.
 *      For details, refer to `whitebox/examples/pyro/BiasedGradWithUniform.ipynb`.
 *
 * 4. Delta distributions:
 *    (i) A model is not allowed to contain Delta distributions. (ii) A guide can contain
 *    Delta distributions, but assume that every `pyro.sample(_, Delta(v))` is replaced
 *    by `v` before starting our analyser.
 *    - (i) is assumed for brevity. We may consider relaxing it.
 *    - (ii) is sound since taking an expectation over Delta(z;v) is the same as
 *      substituting z with v inside the expectation.
 *    - If the substitution in (ii) is not performed, our analysis result could be
 *      unsound. For instance, consider
 *        guide_1 := C; z ~ Delta(theta), and
 *        guide_2 := C; z ~ Delta(theta); if z > 0 then score(1) else score(2).
 *      Suppose that the density of C is differentiable w.r.t. theta. Then,
 *        (a) density of guide_1 after substituting z with theta is differentiable
 *            w.r.t. theta; and
 *        (b) density of guide_2 after substituting z with theta is not differentiable
 *            w.r.t. theta.
 *      Now suppose that we pass guide_{1,2} to our analyser without performing
 *      the substitution in (ii). To make our analyser produce (a), `diff_dk_sig(Delta)`
 *      should be `_,[true]`, not `_,[false]`. However, this makes our analyser conclude
 *      the negation of (b), since the non-differentiability coming from the branch on z
 *      cannot lead to (b) since the data dependency on theta does not flow into z.
 *      In sum, to guarantee the soundness of our analyser, the substitution in (ii)
 *      should be performed before our analyser is called.*)


(** Differentiability-related properties. *)
(* type for properties related to differentiability:
 *   Diff: differentiable w.r.t. some parameters S.
 *   Lips: locally Lipschitz w.r.t. some parameters S.
 *   Top:  always true. *)
(*type diff_prop = Diff | Lips | Top*)

(* order on diff_prop:
 *   We give the following order based on implication:
 *   dp1 <= dp2 <==> for any func f, if f is dp1 w.r.t. S, then f is dp2 w.r.t. S. *)
let diff_prop_leq (dp1: diff_prop) (dp2: diff_prop): bool =
  match dp1, dp2 with
  | _, _ when dp1 = dp2 -> true
  | Diff, Lips | Diff, Top | Lips, Top -> true
  | _ -> false


(** Data for analysis. *)
(* Config vars
 * - dp_goal: property to be analyzed.
 *   - ASSUME: it is not Top and is initialised by `analyze ...`.
 * - debug: global debug. *)
let dp_goal: diff_prop ref = ref Lips
let set_goal = function
  | Top  -> failwith "analyze: dp_goal must not be Top"
  | goal -> dp_goal := goal
let debug: bool ref = ref false
let printf_debug (f: ('a, form, unit) format): 'a =
  if !debug then F.printf f
  else           F.ifprintf F.std_formatter f

(* Utilities for debugging the analysis *)
let dbg_info_sample msg sel pp acc =
  printf_debug "%s statement\n" msg;
  begin
    match sel with
    | None ->
        printf_debug "sel = None\n"
    | Some l ->
        printf_debug "sel != None, |sel| = %d, # of true in sel = %d\n"
          (List.length l)
          (List.length (List.filter (fun b -> b) l))
  end;
  printf_debug "---------- abstract state ----------\n%a" pp acc;
  printf_debug "------------------------------------\n\n"


(* SPECIFICATION: Maps each distribution into a tuple:
 *  1. Boolean indicating whether its density is continuously differentiable
 *     w.r.t. a sampled value over its domain
 *     - This boolean is used to determine which sampled variables are reparameterisable.
 *     - For discrete distributions, the boolean is false, since we cannot
 *       differentiate its density w.r.t. a discrete sampled value.
 *
 *  2. List of booleans indicating whether its density is continuously differentiable
 *     w.r.t. each parameter (of distribution) over its domain
 *     - These booleans are used to determine which sampled variables and
 *       parameters (not of distribution, but to be trained) are reparameterisable.
 *     - For discrete parameters (of distribution), the boolean is false
 *       since we cannot differentiate its density w.r.t. a discrete parameter.
 *
 *  3. NOTE: THIS IS CURRENTLY NOT IN USE.
 *     The subsets of the domains of parameters
 *     (i.e., some values should be positive, and the violation
 *     of this domain constraint may lead to non-differentiability
 *     due to altered control flow).
 *     WL: why is this part commented out? where do we check whether
 *         each parameter is in the supposed domain? E.g., sigma of Normal is in R_{>0},
 *         and a and b of Uniform is in {(a,b) in R^2 : a<b}.
 *
 *  REMARKS:
 *  - Precise definition of continuous differentiability is as follows.
 *    Consider a density f(z; a1, ..., an) of a distribution. Let V_cont \subseteq
 *    {z, a1, ..., an} be the set of continuous variables (i.e., variables whose domain
 *    is a union of non-empty open sets in R^|V_cont|), and V_disc =  {z, a1, ..., an} \ V_cont
 *    be the set of non-continuous variables. Let D_cont and D_disc be the domain of
 *    V_cont and V_disc, respectively.
 *
 *    We give some examples on the domain of a density:
 *    - the density of Normal(z; m, s) has the domain {(z, m, s) \in R x R x R_{>0}};
 *    - the density of Exponential(z; r) has the domain {(z, r) \in R_{>=0} x R_{>0}};
 *    - the density of Uniform(z; a, b) has the domain {(z, a, b) \in R x R x R | a <= z < b};
 *    - the probability mass of Poisson(z; l) has the domain {(z, l) \in Z_{>=0} x R_{>0}};
 *    - the probability mass of Delta(z; v) has the domain {(z, v) \in R x R | z = v}.
 *
 *    For any V \subset V_cont, we say d is continuously differentiable w.r.t. V
 *    iff f[V_disc:v_disc] : D_cont (\subseteq R^|V_cont|) -> R is  continuously
 *    differentiable w.r.t. V for all v_disc \in D_disc. Here f[V_disc:v_disc] denotes
 *    the function obtained from f by fixing the value of V_disc to v_disc.
 *
 *  - We consider continuous differentiability instead of mere differentiability
 *    since the former is variable-monotone but the latter is not.
 *
 *  - Special care is required for boolean values of Uniform and Delta, as the support of
 *    Uniform depends on its parameters, and the support of Delta is a singleton set.
 *    To guarantee the unbiasedness of selective reparameterisation, we put some
 *    ASSUMPTIONS on Uniform and Delta (see the top).
 *
 *  WARNING:
 *  - The distribution constructors are usually applied in the broadcasting mode
 *    during sample and observe. The diff_dk_sig function does not consider the use
 *    of broadcasting. The user of the function should take care of it. *)
let diff_dk_sig = function
  | Normal              -> true , [true ; true ] (* loc, scale *)
  | Exponential         -> true , [true ]        (* rate *)
  | Gamma               -> true , [true ; true ] (* concentration, rate *)
  | Beta                -> true , [true ; true ] (* concentration1, concentration0 *)
  | Dirichlet _         -> true , [true ]        (* concentration *)
  | Poisson             -> false, [true ]        (* rate  *)
  | Categorical _       -> false, [true ]        (* probs *)
  | Bernoulli           -> false, [true ]        (* probs *)
  | OneHotCategorical _ -> false, [true ]        (* probs *)
  | Subsample (_, _)    -> false, [false; false]
  (* Uniform and Delta require special care. *)
  | Uniform _           -> true , [true ; true ] (* low, high *)
  | Delta               -> false, [false]        (* v *)
  (* new. *)
  | LogNormal           -> true , [true ; true ] (* loc, scale *)
  | ZeroInflatedNegativeBinomial -> false, [true; true; true] (* total_count(float), [logits, gate_logits] *)
  | Multinomial         -> false, [false; true ] (* total_count(int), probs *)
  | VonMises            -> true , [true ; true ] (* loc, concentration *)
  | MaskedMixtureGammaDelta -> false, [false; true; true; false] (* mask, gamma_concentration, gamma_rate, delta_value *)
  | MaskedMixtureBetaDelta  -> false, [false; true; true; false] (* mask, beta_concentration1, beta_concentration0, delta_value *)

(* Table storing differentiability information about functions.
 * - None means no information.
 * - Some[b1;b2;...] keeps information about the continuous differentiability of the function.
 *   b_i indicates whether the function is continuously differentiable w.r.t. its i-th argument. *)
let diff_funct_sig v =
  match v with
  (*
   * Functions that return a tensor or float.
   *)
  | "torch.arange"     (* args = ([start,] end [, step]) *)
  | "torch.ones"       (* args = ( *shape ) *)
  | "torch.eye"        (* args = ( n ) *)
  | "torch.rand"       (* args = ( *shape ) *)
  | "torch.randn"      (* args = ( *shape ) *)
  | "torch.zeros"      (* args = ( *shape ) *)
  | "torch.LongTensor" (* args = ( data ). performs rounding operations. *)
    -> Some [ ]

  | "access_with_index"      (* args = (tensor, index) *)
  | "float"                  (* args = (data) *)
  | "torch.cat"              (* args = (tensors[, dim]) *)
  | "torch.cumprod"          (* args = (tensor, dim) *)
  | "torch.exp"              (* args = (tensor) *)
  | "torch.index_select"     (* args = (tensor, dim, index) *)
  | "torch.log"              (* args = (tensor) *) (* TODO: check if arg > 0. *)
  | "torch.reshape"          (* args = (tensor, shape) *)
  | "torch.sigmoid"          (* args = (tensor) *)
  | "torch.squeeze"          (* args = (tensor[, dim]) *)
  | "torch.sum"              (* args = (tensor, dim) *)
  | "torch.tensor"           (* args = (data) *)
  | "torch.transpose"        (* args = (tensor, dim0, dim1) *)
  | "torch.FloatTensor"      (* args = (data) *)
  | "torch.Tensor.clone"     (* args = (tensor) *)
  | "torch.Tensor.detach"    (* args = (tensor) *)
  | "torch.Tensor.expand"    (* args = (tensor, *shape) *)
  | "torch.Tensor.new_ones"  (* args = (tensor, shape) *)
  | "torch.Tensor.new_zeros" (* args = (tensor, shape) *)
  | "torch.Tensor.reshape"   (* args = (tensor, shape) *)
  | "torch.Tensor.transpose" (* args = (tensor, dim0, dim1) *)
  | "torch.Tensor.view"      (* args = (tensor, *shape) *)
  | "F.affine_grid"          (* args = (tensor, shape) *)
  | "F.pad"                  (* args = (tensor, pad[, value]) *)
  | "F.softmax"              (* args = (tensor[, dim]) *)
  | "F.softplus"             (* args = (tensor) *)
  | "TDU.logits_to_probs"    (* args = (tensor[, is_binary]) *)
  | "Vindex"                 (* args = (tensor) *)
    -> Some [ true ]

  | "torch.abs"              (* args = (tensor) *)
  | "torch.max"              (* args = (tensor) *)
  | "F.relu"                 (* args = (tensor) *)
    -> (match !dp_goal with
        | Diff -> Some [ false ]
        | Lips -> Some [ true  ]
        | _    -> failwith "error")

  | "torch.matmul"           (* args = (tensor, tensor) *)
    -> Some [ true; true ]

  | "F.grid_sample"          (* args = (input:tensor, grid:tensor [, mode]) *)
     (* WARNING:
      * - `mode` can be either "bilinear" or "nearest"; its default value is "bilinear".
      * - For `mode`="bilinear":
      *   - for differentiability, `Some[true;false]` is sound, but `Some[true;true]` is unsound;
      *   - for Lipschitzness, `Some[true;true]` is sound.
      * - For `mode`="nearest":
      *   - for both differentiability and Lipschitzness,
      *    `Some[true;false]` is sound, but `Some[true;true]` is unsound.
      * - Hence, when `mode`="nearest" and our analyser checks Lipschitzness,
      *   using `Some[true;true]` could produce unsound analysis results. *)
    -> (match !dp_goal with
        | Diff -> Some [ true; false ]
        | Lips -> Some [ true; true  ]
        | _    -> failwith "error")

  | "update_with_field"      (* args = (src, field_name, new_value) *)
  | "update_with_index"      (* args = (src, indices, new_value) *)
    -> Some [ true; false; true ]

  | "torch.Tensor.scatter_add_" (* args = (dim, index, tensor) *)
    -> Some [ false; false; true ]

  (*
   * Functions that return an object, or receives an object as an argument.
   *
   * Note: If a function returns an object, we say that the object is differentiable with respect to
   *       its parameters. The functions and methods invoked on the object should then revise
   *       this default decision if their outcomes are not differentiable with respect to
   *       the parameters used to create their argument objects. This convention is applied to
   *       our handling of Categorical and Categorical.log_prob.
   * Note: There is only one example that requires the extended notion of differentiability
   *       described above: `whitebox/refact/test/pyro_example/lda_guide2.py`.
   *       Since this example is not included in our final benchmarks, we do not need to consider
   *       this extended notion of differentiability, and using `Some[]` for the following cases
   *       would still produce desired analysis results for our final benchmarks.
   *)
  | "Categorical"          (* args = (tensor) *)
  | "Categorical.log_prob" (* args = (distribution, tensor) *)
  | "OneHotCategorical"          (* args = (tensor) *)
  | "OneHotCategorical.log_prob" (* args = (distribution, tensor) *)
    -> Some [ true ]

  (*
   * All the functions below are handled in the most imprecise manner;
   * it may be possible to very easily improve on them based on their
   * semantics.
   *)
  (* Functions that may return non-tensor and non-float objects. *)
  (* --- python-related *)
  | "dict"
  | "int"
  | "len"
  | "range"
  | "tuple"
  (* --- torch-related*)
  | "nn.BatchNorm1d"
  | "nn.Dropout"
  | "nn.Linear"
  | "nn.LSTMCell"
  | "nn.Parameter"
  | "nn.ReLU"
  | "nn.RNN"
  | "nn.Sigmoid"  (* -> Some [ true ] *)
  | "nn.Softmax"  (* -> Some [ true ] *)
  | "nn.Softplus" (* -> Some [ true ] *)
  | "nn.Tanh"     (* -> Some [ true ] *)
  | "torch.no_grad"
  | "torch.unbind"
  | "torch.Tensor.long"
  | "torch.Tensor.size"
  | "RYLY[constraints.positive]"
  | "RYLY[constraints.negative]"
  (* --- pyro-related*)
  | "pyro.markov"
  | "pyro.plate"
  | "pyro.poutine.mask"
  | "pyro.poutine.scale"
  | "pyro.util.set_rng_seed"
  | "PDU.broadcast_shape"
  | "Dirichlet"
  | "Dirichlet.sample"
  | "LogNormal"
  | "LogNormal.sample"
  | "Uniform"
  | "Uniform.sample"
  (* --- our ir-related *)
  | "RYLY"
    (* nothing known; will assume non differentiable in all args *)
    -> Some [ ]

  (* Method calls *)
  | "decoder_fst.bias.data.normal_"
  | "decoder_fst.weight.data.normal_"
  | "encoder_y_fst.bias.data.normal_"
  | "encoder_y_fst.weight.data.normal_"
  | "encoder_z_fst.bias.data.normal_"
  | "encoder_z_fst.weight.data.normal_"
  | "layer1.bias.data.normal_"
  | "layer1.weight.data.normal_"
  | "layer2.bias.data.normal_"
  | "layer2.weight.data.normal_"
  | "layer3.bias.data.normal_"
  | "layer3.weight.data.normal_"
  | "z_pres.append"
  | "z_where.append"
    (* nothing known; will assume non differentiable in all args *)
    -> Some [ ]

  (* All the rest *)
  | _
    -> F.printf "TODO,function: %S\n" v; None

(* Table storing differentiability information about function-returning function.
 * - None means no information.
 * - Some(b,l) keeps information about the continuous differentiability of
 *   the returned function. b indicates whether the function is continuously differentiable
 *   w.r.t. all implicit parameters kept by the function (if such parameters exist).
 *   l stores continuous differentiability information w.r.t. the function arguments. *)
let diff_functgen_sig f args kwargs =
  match f with
  | "nn.BatchNorm1d"
  | "nn.Dropout" (* Dropout is differentiable when mask is assumed to be fixed. *)
  | "nn.Linear"
  | "nn.Sigmoid"
  | "nn.Softmax"
  | "nn.Softplus"
  | "nn.Tanh"
    -> Some (true, [true])
  | "nn.ReLU"
    -> (match !dp_goal with
        | Diff -> Some (true, [false])
        | Lips -> Some (true, [true ])
        | _    -> failwith "error")
  | "nn.LSTMCell"
    -> Some (true, [true; true])
  | "nn.RNN"
    -> begin
      try
        match (List.find (fun x -> (fst x) = Some "nonlinearity") kwargs) with
        | _, Str "tanh"
          -> Some (true, [true; true])
        | _, Str "relu"
          -> (match !dp_goal with
              | Diff -> Some (false, [false; false])
              | Lips -> Some (true , [true ; true ])
              | _    -> failwith "error")
        | _ -> failwith "diff_functgen_sig, nn.RNN: unreachable."
      with
      | Not_found (* Same as "tanh" since default "nonlinearity" is "tanh". *)
        -> Some (true, [true; true])
    end
  | _
    -> None


(* WL: Checked up to here. *)
(** Computation for analysis. *)
module Make =
  functor (DN: DOM_NUM) -> functor (FD: DOM_DIFF_FWD) -> functor (CD: DOM_DIFF_COMP) ->
  struct
    (** Parameterization *)
    (* Should the analysis do safety checks *)
    let ref_do_safe_check = ref true
    (* Already available safety information *)
    let ref_safety_info: SI.t ref = ref SI.top

    (* Types for objects, such as constraint objects.
     * These objects should behave as if they were pure (i.e. update-free)
     * objects from the perspective of the analysis. This means that they
     * are indeed pure objects, or the information recorded in a type
     * is invariant with respect to all possible updates. *)
    type obj_t =
      | O_Constr of constr
      | O_Dist of string
      | O_Fun of string
      | O_Nil

    let pp_obj_t chan = function
      | O_Constr(c) ->  Printf.fprintf chan "ConstrObj[%a]" pp_constr c
      | O_Dist(d) ->  Printf.fprintf chan "DistObj[%s]" d
      | O_Fun(f) ->  Printf.fprintf chan "FunObj[%s]" f
      | O_Nil ->  Printf.fprintf chan "NoneObj"
    let fp_obj_t fmt = function
      | O_Constr(c) ->  F.fprintf fmt "ConstrObj[%a]" fp_constr c
      | O_Dist(d) ->  F.fprintf fmt "DistObj[%s]" d
      | O_Fun(f) ->  F.fprintf fmt "FunObj[%s]" f
      | O_Nil ->  F.fprintf fmt "NoneObj"

    type t =
        { (* "Parameters":
           * Parameters with respect to which we track differentiability *)
          t_pars:    SS.t ;
          (* Forward computed diff information *)
          t_fdiff:   FD.t ;
          (* "Variable-Function-Information":
           * Maps each variable to information about a function object
           * stored in the variable.
           * All unmapped variables are implicitly mapped to top, the lack
           * of any information. (* WL: is there top here? *)
           * The stored information is a pair of a boolean and a list.
           * The boolean describes the differentiability with respect to
           * implicit parameters to the function object, if such
           * parameters exist. Thus, if it is true, there are no implicit
           * parameters or the function is differentiable with respect to
           * such parameters. *)
          (* WL: More details. (b,l) ==>
           *   b = 1_[f is differentiable wrt all implicit params].
           *   l_i = 1_[f is differentiable wrt ith arg]. *)
          t_vfinfo:  (bool * bool list) SM.t ;
          (* "Variable-Object-Information":
           * Maps each variable to information about the object stored in the
           * variable. *)
          t_voinfo:  obj_t SM.t ;
          (* Whether the program is runtime errors (RTE) free in numerics
           * expressions, typically due to divisions *)
          t_safety:  SI.t;
          (* "Variable-Numerical predicates":
           * Numerical predicates over variables expressed in parameter
           * domain DN.
           * IMPORTANT: If a variable is bounded to Top, it is numeric. *)
          t_vnum:    DN.t ;
          (* Addition of compositional analysis support *)
          t_cdiff:   CD.t ;
        }
    let get_d_ndiff (t: t): SS.t = FD.get_d_ndiff t.t_fdiff
    let get_d_diff (t: t): SS.t  = FD.get_d_diff t.t_pars t.t_fdiff

    let fp (ind: string) fmt t =
      let nind = ind ^ "  " in
      let fpm_fi = fpm (fp_pair fp_bool (fp_list fp_bool)) in
      let fpm_oi = fpm fp_obj_t in
      F.fprintf fmt "%spars: %a\n%a%svfinfo:\n%a%svoinfo:\n%a%s%s:\n%a"
        ind ss_fp         t.t_pars
        (FD.fp nind)      t.t_fdiff
        ind fpm_fi        t.t_vfinfo
        ind fpm_oi        t.t_voinfo
        ind DN.name (DN.fp nind) t.t_vnum;
      F.fprintf fmt "%scompositional information:\n%a%ssafety info:%a\n"
        ind (CD.fp nind) t.t_cdiff
        ind SI.fp t.t_safety

    (* Wrappers for some numerical domain operations *)
    let is_bot t = DN.is_bot t.t_vnum
    let guard e t = { t with t_vnum = DN.guard e t.t_vnum }

    (* Lattice operations *)
    let t_union (acc0: t) (acc1: t): t =
      let info_map_join m0 m1 =
        map_join_inter
          (fun c0 c1 ->
            if c0 = c1 then Some c0
            else None) m0 m1 in
      { t_pars    = SS.union      acc0.t_pars    acc1.t_pars;
        t_fdiff   = FD.join       acc0.t_fdiff   acc1.t_fdiff;
        t_vfinfo  = info_map_join acc0.t_vfinfo  acc1.t_vfinfo;
        t_voinfo  = info_map_join acc0.t_voinfo  acc1.t_voinfo;
        t_vnum    = DN.join       acc0.t_vnum    acc1.t_vnum;
        t_cdiff   = CD.join       acc0.t_cdiff   acc1.t_cdiff;
        t_safety  = SI.join       acc0.t_safety  acc1.t_safety }

    let t_equal (acc0: t) (acc1: t): bool =
      let info_map_equal m0 m1 =
        map_equal (fun v i -> true) (fun p1 p2 -> p1 = p2) m0 m1 in
      SS.equal            acc0.t_pars    acc1.t_pars
        && FD.equal       acc0.t_fdiff   acc1.t_fdiff
        && info_map_equal acc0.t_vfinfo  acc1.t_vfinfo
        && info_map_equal acc0.t_voinfo  acc1.t_voinfo
        && DN.equal       acc0.t_vnum    acc1.t_vnum
        && CD.equal       acc0.t_cdiff   acc1.t_cdiff
        && SI.equal       acc0.t_safety  acc1.t_safety

    (** Display of analysis results *)
    let fp_result ind fmt (abs: t): unit =
      let nind = ind ^ "  " in
      F.fprintf fmt "%a%sFunctions bound to variables:\n"
        (FD.fp_results ind) abs.t_fdiff ind;
      SM.iter
        (fun v fi ->
          F.fprintf fmt "\t%-30s\t=>\t%a\n" v (fp_pair fp_bool (fp_list fp_bool)) fi
        ) abs.t_vfinfo;
      F.fprintf fmt "%sObjects bound to variables:\n" ind;
      SM.iter
        (fun v oi ->
          F.fprintf fmt "\t%-30s\t=>\t%a\n" v fp_obj_t oi
        ) abs.t_voinfo;
      F.fprintf fmt "%sNumerical information<%s>:\n%a" ind
        DN.name (DN.fp nind) abs.t_vnum;
      let sgn = if SI.nowhere_div0 abs.t_safety then "" else "NOT " in
      F.fprintf fmt "%s%sProved free of runtime errors\n" ind sgn

    (** Analysis of expression *)
    type texp =
        { (* Parameters on which the expression may depend on *)
          te_pdep:   SS.t ;
          (* Parameters with respect to which the expression may depend on
           * and may be non-differentiable:
           * => in general it is always sound to make this field equal to
           *    te_pdep; *)
          te_pndiff: SS.t }

    (* Table storing information about object-returning function.
     * - None means no information.
     * - Some ot keeps information about the returned object *)
    let funct_obj_sig f args kwargs =
      match f with
      | "RYLY[constraints.positive]" ->
          Some(O_Constr(C_Pos))
      | "RYLY[constraints.negative]" ->
          Some(O_Constr(C_Neg))
      | "Categorical"
      | "OneHotCategorical" ->
          Some(O_Dist(f))
      (* see diff_funct_sig above. *)
      | "nn.BatchNorm1d"
      | "nn.Dropout"
      | "nn.Linear"
      | "nn.LSTMCell"
      | "nn.ReLU"
      | "nn.RNN"
      | "nn.Sigmoid"
      | "nn.Softmax"
      | "nn.Softplus"
      | "nn.Tanh" ->
          Some(O_Fun(f))
      | _ ->
          None



    (** Utility functions *)
    let accumulate_guard_pars (accu: SS.t) (del: texp list)
        (ok_el: bool list option): SS.t =
      let default () =
        List.fold_left (fun acc de -> SS.union acc de.te_pdep) accu del in
      match ok_el with
      | None -> default ()
      | Some ok_el ->
          try
            let f acc ok de = if ok then acc else SS.union acc de.te_pdep in
            List.fold_left2 f accu ok_el del
          with Invalid_argument _ -> default ()

    let imply (acc: t) (e_in: expr): bool =
      let rec simplify e =
        match e with
        | Name x ->
            begin
              try
                match (SM.find x acc.t_voinfo) with
                | O_Nil -> Nil
                | _ -> e
              with Not_found -> e
            end
        | UOp (uop, e0) ->
            let e0_sim = simplify e0 in
            UOp (uop, e0_sim)
        | BOp (bop, e0, e1) ->
            let e0_sim = simplify e0 in
            let e1_sim = simplify e1 in
            BOp (bop, e0_sim, e1_sim)
        | Comp (cop, e0, e1) ->
            let e0_sim = simplify e0 in
            let e1_sim = simplify e1 in
            Comp (cop, e0_sim, e1_sim)
        | List es0 ->
            List (List.map simplify es0)
        | _ ->
            e in
      let r = DN.imply acc.t_vnum (simplify e_in) in
      if false then F.printf "imply => %b\n" r;
      r

    (* Check that an expression will not result in a runtime error *)
    let rte_check_expr loc (acc: t) (e: expr): bool =
      let module M = struct exception Stop end in
      let rec aux = function
        | Nil | True | False | Num _ | Str _ | Name _ -> ( )
        | UOp (_, e) -> aux e
        | BOp (op, e0, e1) ->
            aux e0;
            aux e1;
            if op = Div then
              if not (SI.no_div0 !ref_safety_info loc)
                  && not (imply acc (Comp (NotEq, e1, Num (Float 0.)))) then
                begin
                  (*F.printf "Possible division by 0\n";*)
                  raise M.Stop
                end
              else ()
            else if List.mem op [Add; Sub; Mult; Pow; And; Or] then ()
            else raise M.Stop
        | List el | StrFmt (_, el) -> List.iter aux el
        | Comp (_, e0, e1) ->
            aux e0;
            aux e1
        | _ -> raise M.Stop in
      let b =
        try aux e; true
        with M.Stop -> false
      in
      if false then F.printf "Safety check %a => %b\n" fp_expr e b;
      b

    let obj_expr (m : obj_t SM.t) = function
      | Name x ->
          begin
            try Some(SM.find x m) with Not_found -> None
          end
      | _ -> None

    let ndpars_call_args (acc: t) (fsig: bool list) (del: texp list): SS.t =
      let rec aux (fsig: bool list) (del: texp list): SS.t =
        match fsig, del with
        | diffarg :: fsig, d :: del ->
            let pn = aux fsig del in
            let pndiff =
              if diffarg then d.te_pndiff
              else d.te_pdep in
            SS.union pn pndiff
        | _ :: _, [ ] ->
            (* May not be differentiable at all *)
            acc.t_pars
        | [ ], d :: del ->
            let pn = aux [ ] del in
            SS.union pn d.te_pdep
        | [ ], [ ] ->
            SS.empty in
      aux fsig del

    let update_info_map
          (tbl: string -> expr list -> keyword list -> 'a option)
          (m: 'a SM.t) (x: string) (f: string)
          (args: expr list) (kwargs: keyword list): 'a SM.t =
      match tbl f args kwargs with
      | Some info -> SM.add x info m
      | None ->
          begin
            match f, args with
            | "update_with_field", Name y :: _ ->
                begin
                  try SM.add x (SM.find y m) m with Not_found -> SM.remove x m
                end
            | _ -> SM.remove x m
          end
    let update_finfo_map = update_info_map diff_functgen_sig
    let update_oinfo_map = update_info_map funct_obj_sig

    let has_no_obs (acc: t) = function
      | None -> true
      | Some o ->
          match (obj_expr acc.t_voinfo o) with
          | Some O_Nil -> true
          | _ -> false

    let get_obs (acc: t) obs_opt =
      let err_msg =
        "Should not be reached: sample statement confused as observe" in
      match obs_opt with
      | None -> failwith err_msg
      | Some o ->
          match (obj_expr acc.t_voinfo o) with
          | Some O_Nil -> failwith err_msg
          | Some _ | None -> o

    let do_acmd loc (acc: t) (ac: acmd): t =
      match ac with
      | AssnCall (_, Name "pyro.param", Str pname :: Name x :: _, kargs)
      | AssnCall (x, Name "pyro.param", Str pname :: _, kargs) ->
          let vnum =
            try
              let (_, e) =
                List.find (fun (k,_) -> k = Some("constraint")) kargs in
              match (obj_expr acc.t_voinfo e) with
              | Some (O_Constr(constr)) ->
                  (*F.printf "constraint!\n";*)
                  DN.heavoc x constr acc.t_vnum
              | None | Some (O_Dist _) | Some (O_Fun _) | Some O_Nil ->
                  DN.heavoc x C_Num acc.t_vnum
            with Not_found ->
              (*F.printf "pyro.param: not found\n";*)
              DN.heavoc x C_Num acc.t_vnum in
          { acc with
            t_fdiff   = FD.pyro_param x pname acc.t_fdiff;
            t_pars    = SS.add pname acc.t_pars;
            t_vfinfo  = SM.remove x acc.t_vfinfo;
            t_voinfo  = SM.remove x acc.t_voinfo;
            t_vnum    = vnum;
            t_cdiff   = CD.pyro_param x pname }
      | AssnCall (_, Name "pyro.param", _, _) ->
          F.printf "unbound-pyro.param: %a\n" fp_acmd ac;
          failwith "unbound-pyro.param"
      | AssnCall (_, Name "pyro.module", Str pname :: Name x :: _, _ ) ->
          let isdiff, _ = lookup_with_default x acc.t_vfinfo (false,[]) in
          { acc with
            t_pars  = SS.add pname acc.t_pars;
            t_fdiff = FD.pyro_module x (pname, isdiff) acc.t_fdiff;
            t_cdiff = CD.pyro_module x (pname, isdiff) }
      | AssnCall (_, Name "pyro.module", Str pname :: _, _ ) ->
          { acc with
            t_pars  = SS.add pname acc.t_pars;
            t_fdiff = FD.pars_add pname acc.t_fdiff;
            t_cdiff = acc.t_cdiff }
      | AssnCall (_, Name "pyro.module", _, _) ->
          failwith "unbound-pyro.module"
      | AssnCall (x, Name v, el, kwargs) ->
          (* Assumption: all primitives always terminate *)
          (* Check if the return value is differentiable *)
          (*F.printf "%s = call %s( ... )\n" x v;*)
          let is_ipdiff, ofsig =
            try
              let is_ipdiff, fsig = SM.find v acc.t_vfinfo in
              is_ipdiff, Some fsig
            with Not_found ->
              let ofsig = diff_funct_sig v in
              ofsig != None, ofsig in
          (* Commpositional abstraction *)
          let fdiff =
            let safe_no_div0 i e =
              let loc = Loc.parn loc i in
              SI.no_div0 !ref_safety_info loc ||
              imply acc (Comp (NotEq, e, Num (Float 0.))) in
            FD.call ~safe_no_div0 (x, v, el) (is_ipdiff, ofsig) acc.t_fdiff in
          (* Compositional differentiability information *)
          let cdiff =
            match is_ipdiff, ofsig with
            | _    , Some fsig -> CD.call v (is_ipdiff, fsig) (Some x) (Nil, el)
            | false, None -> CD.error "assigncall, other, not found either"
            | true , None -> CD.error "assigncall, contradiction" in
          (* Check if the return value is a function with
           * known information about its differentiability *)
          let finfo_map = update_finfo_map acc.t_vfinfo x v el kwargs in
          let oinfo_map = update_oinfo_map acc.t_voinfo x v el kwargs in
          let num =
            match obj_expr acc.t_voinfo (Name v) with
            | None | Some (O_Constr _) | Some (O_Dist _) | Some O_Nil ->
               DN.call_prim x v el acc.t_vnum
            | Some (O_Fun(f)) ->
               DN.call_obj x f el acc.t_vnum in
          (* Checking the absence of errors *)
          let safe, _ =
            List.fold_left
              (fun (safe, i) e ->
                let loc = Loc.parn loc i in
                let b = rte_check_expr loc acc e in
                SI.acc_check_div safe loc b, i + 1
              ) (acc.t_safety, 0) el in
          { acc with
            t_fdiff   = fdiff;
            t_vfinfo  = finfo_map;
            t_voinfo  = oinfo_map;
            t_vnum    = num;
            t_safety  = safe;
            t_cdiff   = cdiff }
      | AssnCall (_, _, _, _) ->
          F.printf "TODO,complex assncall: %a\n" fp_acmd ac;
          { acc with t_cdiff = CD.error "assigncall,complex case" }
      | Assert _
      | Assume _ ->
          { acc with t_cdiff = CD.id () }
      | Assn (x, e) ->
          (* Mark whether x may depend on paramters *)
          let oinfo =
            match e with
            | Nil -> SM.add x O_Nil acc.t_voinfo
            | _ -> SM.remove x acc.t_voinfo in
          (* Checking the absence of errors *)
          let chk = rte_check_expr loc acc e in
          (* Getting safety information *)
          let fdiff =
            let safe_no_div0 e =
              SI.no_div0 !ref_safety_info loc ||
              imply acc (Comp (NotEq, e, Num (Float 0.))) in
            FD.assign ~safe_no_div0:safe_no_div0 x e acc.t_fdiff in
          { acc with
            t_fdiff   = fdiff;
            t_vfinfo  = SM.remove x acc.t_vfinfo;
            t_voinfo  = oinfo;
            t_vnum    = DN.assign x e acc.t_vnum;
            t_safety  = SI.acc_check_div acc.t_safety loc chk;
            t_cdiff   = CD.assign ~safe:(SI.no_div0 !ref_safety_info loc) x e }
      | Sample (x (* x *), n (* S *), d (* Distr *), a (* E1, E2 *),
                o_opt (* Obs *), repar)
        when has_no_obs acc o_opt ->
          let parname =
            let dbg = false in
            match n with
            | Str s -> s
            | StrFmt (s, _) ->
                if dbg then
                  F.printf "TODO,Sample formatter: %S\n" s;
                s
            | _ ->
                if dbg then
                  F.printf "unbound parameter expression: %a\n" fp_expr n;
                failwith "unbound" in
          (* Non-differentiability information about current distribution *)
          let dist_diff, fsig_diff = diff_dk_sig (fst d) in
          (* Arguments *)
          let sel = DN.check_dist_pars (fst d) a acc.t_vnum in
          (* Checking the absence of errors *)
          let safe =
            SI.acc_check_div acc.t_safety loc (rte_check_expr loc acc n) in
          let safe, _ =
            let ssel =
              match sel with
              | None -> List.map (fun _ -> false) a
              | Some l -> l in
            List.fold_left2
              (fun (safe, i) ex pok ->
                let loc = Loc.parn loc i in
                let b = rte_check_expr loc acc ex in
                let safe = SI.acc_check_div safe loc b in
                let safe = SI.acc_par_ok safe loc pok in
                safe, i + 1
              ) (safe, 0) a ssel in
          (* Forward differentiability information *)
          let fdiff =
            let safe_no_div0 io e =
              let loc =
                match io with
                | None -> loc
                | Some i -> Loc.parn loc i in
              SI.no_div0 !ref_safety_info loc ||
              imply acc (Comp (NotEq, e, Num (Float 0.))) in
            FD.pyro_sample
              ~safe_no_div0:safe_no_div0
              (d, dist_diff, fsig_diff) (x, n, a, parname) sel acc.t_fdiff; in
          (* Compositional differentiability information *)
          let cdiff =
            let fdiv0 i = SI.no_div0 !ref_safety_info (Loc.parn loc i) in
            let fparok i = SI.par_ok !ref_safety_info (Loc.parn loc i) in
            let expok = SI.no_div0 !ref_safety_info loc in
            CD.pyro_sample
              ~safei:(fdiv0,fparok,expok) ~repar
              x parname (dist_diff, fsig_diff) (n, a) in
          { t_pars    = SS.add parname acc.t_pars;
            t_fdiff   = fdiff;
            t_vfinfo  = SM.remove x acc.t_vfinfo;
            t_voinfo  = SM.remove x acc.t_voinfo;
            t_vnum    = DN.sample x d acc.t_vnum;
            t_safety  = safe;
            t_cdiff   = cdiff }
      | Sample (_ (* ? *), _n (* S *), d, a (* E1, E2 *), o_opt (* E0 *), o_rep) ->
          (* XR: for an observation, I do not think that x should change;
           *     thus we should not lose precision in it/modify it *)
          (* HY: This part is unsound. The sample statement becomes observe, when it
           *     is not the case that o_opt = Some o and o represents None in Python.
           *     Currently, we assume that the object tracking part is 100% accurate
           *     as far as this check is concerned, so that we can detect the case
           *     simply by checking o_opt is Some o with o != O_Nil. *)
          if o_rep then
            failwith "ERROR: observe statement should not be reparameterised";
          let o = get_obs acc o_opt in
          let dist_diff, fsig_diff = diff_dk_sig (fst d) in
          (* Arguments *)
          let sel = DN.check_dist_pars (fst d) a acc.t_vnum in
          (* Checking the absence of errors *)
          let safe =
            SI.acc_check_div acc.t_safety loc (rte_check_expr loc acc o) in
          let safe, _ =
            let ssel =
              (*F.printf "check dist pars diff none ? %b\n" (sel != None);*)
              match sel with
              | None -> List.map (fun _ -> false) a
              | Some l ->
                  (*List.iter (fun b -> F.printf "%b;" b) l; F.printf "\n";*)
                  l in
            List.fold_left2
              (fun (safe, i) ex pok ->
                let loc = Loc.parn loc i in
                let b = rte_check_expr loc acc ex in
                let safe = SI.acc_check_div safe loc b in
                let safe = SI.acc_par_ok safe loc pok in
                safe, i + 1
              ) (safe, 0) a ssel in
          let cdiff =
            let fdiv0 i = SI.no_div0 !ref_safety_info (Loc.parn loc i) in
            let fparok i = SI.par_ok !ref_safety_info (Loc.parn loc i) in
            let ediv0 = SI.no_div0 !ref_safety_info loc in
            CD.pyro_observe
              ~safei:(fdiv0,fparok,ediv0)
              (dist_diff, fsig_diff) a o in
          { acc with
            t_fdiff   = FD.pyro_observe (dist_diff, fsig_diff) (a, o, sel) acc.t_fdiff ;
            t_safety  = safe;
            t_cdiff   = cdiff }
    let rec do_stmt loc acc com =
      if false then
        F.printf "iter %a\n%a\n" Loc.fp loc (DN.fp "  ") acc.t_vnum;
      match com with
      | Atomic ac ->
          let msg =
            F.asprintf "[Semantics-less placeholder: %s]" (acmd_to_string ac) in
          do_acmd loc { acc with t_cdiff = CD.error msg } ac
      | If (e, b0, b1) ->
          let cgpars = FD.guard_pars_get acc.t_fdiff in
          (*let d = diff_expr loc acc e in*)
          let not_e = (UOp (Not, e)) in
          (*let acc = { acc with t_gpars = SS.union acc.t_gpars d.te_pdep } in*)
          let acc = { acc with t_fdiff = FD.guard_pars_condition e acc.t_fdiff } in
          let acc0 = guard e acc in
          let acc1 = guard not_e acc in
          let tloc = Loc.if_t loc and floc = Loc.if_f loc in
          F.printf "IF: (%b, %b), (%b, %b)\n"
            (is_bot acc0) (imply acc not_e)
            (is_bot acc1) (imply acc e);
          if (is_bot acc0 || imply acc not_e) then
            do_block floc acc1 b1
          else if (is_bot acc1 || imply acc e) then
            do_block tloc acc0 b0
          else
            let acc0 = do_block tloc acc0 b0 in
            let acc1 = do_block floc acc1 b1 in
            let gdep = dep_expr e in
            if not (SS.equal acc0.t_pars acc1.t_pars) then
              printf_debug
                "IF: branches disagree on parameters (ok if constant guard)\n";
            let acc0 = { acc0 with t_cdiff = CD.condition gdep acc0.t_cdiff }
            and acc1 = { acc1 with t_cdiff = CD.condition gdep acc1.t_cdiff } in
            let acc = t_union acc0 acc1 in
            if false then
              F.printf "IF: merge\n%a%a%a" (CD.fp "  ") acc0.t_cdiff
                (CD.fp "  ") acc1.t_cdiff (CD.fp " ") acc.t_cdiff;
            { acc with t_fdiff = FD.guard_pars_set cgpars acc.t_fdiff }
      | For (Name v, e, b0) ->
          FD.check_no_dep e acc.t_fdiff;
          let gdep = SS.add v (dep_expr e) in
          let lloc = Loc.loop loc in
          let rec iter i accv accin =
            flush stdout;
            (*let accout = do_stmt (do_block (guard test accin) b0) inc in*)
            let accout = do_block lloc accin b0 in
            let accout = { accout with
                           t_cdiff = CD.condition gdep accout.t_cdiff } in
            let accout = { accout with
                           t_cdiff = CD.loop_condition gdep accout.t_cdiff } in
            let accj = t_union accv accout in
            if dbg_loop then
              F.printf "iter (%d,%b):\n%a\n" i (t_equal accv accj)
                (fp "  ") accout;
            if t_equal accv accj then accj
            else iter (i+1) accj accout in
          (*guard (UOp (Not, test)) (iter acc acc)*)
          let r = iter 0 acc acc in
          if dbg_loop then F.printf "loop out:\n%a\n" (fp "  ") r;
          r
      | For (_, _, b0) ->
          failwith "TODO,for,other index\n";
      | While (_, b0) ->
          failwith "TODO,while\n";
      | With (l, b0) ->
          (* In state checking analysis, verify no differentiation parameter is used *)
          if !ref_do_safe_check then FD.check_pyro_with l acc.t_fdiff;
          do_block (Loc.next loc) acc b0
      | Break | Continue ->
          (* For now, we do not handle these;
           * serious reasoning over complex control flow needed *)
          failwith "TODO,break/continue\n"
    and do_block loc acc bl =
      snd
        (List.fold_left
           (fun (loc, acc) s ->
             let anxt = do_stmt loc { acc with t_cdiff = CD.id () } s in
             let lnxt = Loc.next loc in
             lnxt, { anxt with t_cdiff = CD.compose acc.t_cdiff anxt.t_cdiff }
           ) (loc, acc) bl)

    let diff_params
        ~(goal: diff_prop)            (* analysis goal *)
        ?(silent: bool = false)       (* if true, reduces a lot outputs *)
        ?(do_safe_check: bool = true) (* whether to do checks for safety *)
        ?(safety_info: SI.t = SI.top) (* pre-computed safety information *)
        ?(analysis_name: string = "")
        ?(allpars: SS.t = SS.empty) (* whether the set of all params is known *)
        ir =
      F.printf "Analysis starts(%s)\n%sAnalysis log...\n" analysis_name sep_2;
      ref_do_safe_check := do_safe_check;
      ref_safety_info := safety_info;
      (* Domain initialisation *)
      begin
        let vars = prog_varparams ~getvars:true ~getpars:false ir in
        DN.init_domain ~vars:vars;
        FD.init_domain ~goal;
        CD.init_domain ~goal ~params:allpars ~vars:vars
      end;
      let res = do_block Loc.start { t_pars    = allpars ;
                                     t_fdiff   = FD.id ;
                                     t_vfinfo  = SM.empty ;
                                     t_voinfo  = SM.empty ;
                                     t_vnum    = DN.top ();
                                     t_cdiff   = CD.id ();
                                     t_safety  = SI.id } ir in
      if silent then
        F.printf "%sAnalysis completed(%s)\n" sep_2 analysis_name
      else
        begin
          F.printf "%sAnalysis output(%s): %a\n" sep_2 analysis_name
            SI.fp res.t_safety;
          F.printf "%a%s" (fp "  ") res sep_2
        end;
      res
  end


(** Functor only for compositional analysis
 **  => suits well pipelined execution, with first non compositional analysis
 **     and then compositional analysis in the second phase
 **  => makes the assumption that some state analysis has been done to rule
 **     out basic errors
 **  => takes as argument the definition of all parameters
 **)
module MakeComp = functor (CD: DOM_DIFF_COMP) ->
  struct
    type t = CD.t

    let do_atomic (ac: acmd): t =
      match ac with
      | AssnCall (_, Name "pyro.param", Str pname :: Name x :: _, kargs)
      | AssnCall (x, Name "pyro.param", Str pname :: _, kargs) ->
          CD.error "call pyro.param"
      | AssnCall (_, Name "pyro.param", _, _) ->
          F.printf "unbound-pyro.param: %a\n" fp_acmd ac;
          failwith "Compositional analysis: unbound-pyro.param"
      | AssnCall (_, Name "pyro.module", Str pname :: Name x :: _, _ ) ->
          CD.error "call pyro.module"
      | AssnCall (_, Name "pyro.module", Str pname :: _, _ ) ->
          CD.error "call pyro.module"
      | AssnCall (_, Name "pyro.module", _, _) ->
          F.printf "unbound-pyro.module: %a\n" fp_acmd ac;
          failwith "Compositional analysis: unbound-pyro.module"
      | AssnCall (x, Name v, el, kwargs) ->
          CD.error "call: other"
      | AssnCall (_, _, _, _) ->
          failwith "Compositional analysis: complex call"
      | Assert _
      | Assume _ ->
          CD.id ()
      | Assn (x, e) ->
          CD.assign x e
      | Sample (x (* x *), n (* S *), d (* Distr *), a (* E1, E2 *), o_opt (* Obs *), _)
        when (*has_no_obs acc o_opt*)true ->
          CD.error "sample"
      | Sample (_ (* ? *), n (* S *), d, a (* E1, E2 *), o_opt (* E0 *), _) ->
          CD.error "observe"

    let rec do_stmt (s: stmt): t =
      match s with
      | Atomic ac -> do_atomic ac
      | If (e, b0, b1) ->
          CD.error "if"
      | For (Name v, e, b0) ->
          CD.error "for"
      | With (l, b) ->
          CD.error "with"
      | For (_, _, _) ->
          failwith "Compositional analysis: complex for loop"
      | While (_, _) | Break | Continue ->
          failwith "Compositional analysis: while,break,continue"
    and do_block (ir: block): t =
      List.fold_left
        (fun acc s ->
          CD.compose acc (do_stmt s)
        ) (CD.id ()) ir
    let diff_params
        ~(goal: diff_prop)
        ?(allpars: SS.t = SS.empty) (* whether the set of all params is known *)
        ir =
      if allpars != SS.empty then
        begin
          let vars = prog_varparams ~getvars:true ~getpars:false ir in
          CD.init_domain ~goal ~params:allpars ~vars:vars
        end;
      let res = do_block ir in
      F.printf "Compositional analysis result:\n%a" (CD.fp "  ") res;
      res
  end


(** Analysis main function wrapper
 ** Inputs:
 ** - a domain
 ** - a file name
 ** Outputs:
 ** - set of parameters for which density is differentiable *)
let analyze
    (dnum: ad_num) (goal: diff_prop)
    ~(title: string) (* title of the analysis (guide/model/...) *)
    ~(flag_fwd: bool)  (* whether to do the forward diff analysis *)
    ~(flag_comp: bool) (* whether to do the forward comp analysis *)
    ~(flag_old_comp: bool) (* whether to use old comp repr *)
    ~(flag_pyast: bool)
    ~(inline_fns: string list)
    ~(input: (string, string * Ir_sig.prog) Either.t)
    (verbose: bool): diff_info =
  debug := verbose;
  (* generate SA *)
  let mod_num =
    let dnum = AD_box in (* Box + signs for all *)
    match dnum with
    | AD_sgn -> (module DN_signs: DOM_NUM)
    | AD_box -> (module DNP_box:  DOM_NUM)
    | AD_oct -> (module DNP_oct:  DOM_NUM)
    | AD_pol -> (module DNP_pol:  DOM_NUM) in
  let module DN = (val mod_num: DOM_NUM) in
  let mod_fwd =
    if flag_fwd then (module FD_standard: DOM_DIFF_FWD)
    else             (module FD_none:     DOM_DIFF_FWD) in
  let module FD = (val mod_fwd: DOM_DIFF_FWD) in
  let module SA = Make( DN )( FD )( CD_none ) in
  let name =
    match input with
    | Either.Left f -> f
    | Either.Right (f, _) -> f in
  (* generate ir by parsing file f. *)
  printf_debug "%sAnalysing %a %S\n%s" sep_1 fp_for_diff_prop goal name sep_2;
  Ir_parse.output := false ;
  let ir =
    match input with
    | Either.Left f ->
          Ir_parse.parse_code ~use_pyast:flag_pyast ~inline_fns None (AI_pyfile f)
          |> Ir_util.simplify_delta_prog
    | Either.Right (f, ir) -> ir in
  F.printf(*_debug*) "Program%s:\n%a%s" title (fpi_block "    ") ir sep_2;
  (* analyse property dp_goal of ir. *)
  set_goal goal;
  (* Launch the analysis *)
  let abs =
    let name = F.asprintf "forward,num:%s" DN.name in
    SA.diff_params ~goal
      ~do_safe_check:true
      ~analysis_name:name ir in
  (* Display analysis output *)
  if !debug then
    F.printf "%sOutput of initial forward analysis:\n%a%s" sep_2
      (SA.fp_result "  ") abs sep_2;
  (* Get important elements out of the analysis *)
  let f_diff = SA.get_d_diff abs
  and f_ndiff = SA.get_d_ndiff abs in
  (* Run the compositional analysis and compare the results *)
  if flag_comp then
    begin
      let mod_cd =
        if flag_old_comp then (module CD_diff:  DOM_DIFF_COMP)
        else                  (module CD_ndiff: DOM_DIFF_COMP) in
      let module CD = (val mod_cd: DOM_DIFF_COMP) in
      let module SAC = Make( DN_none )( FD_none (*FD_standard*) )( CD ) in
      (* also launch experimental simplified compositional analysis engine *)
      let r =
        let name = F.asprintf "forward+relational,num:%s" DN_none.name in
        SAC.diff_params ~goal
          ~do_safe_check:false
          ~safety_info:abs.t_safety
          ~analysis_name:name
          ~allpars:abs.t_pars ir in
      let rd = r.t_cdiff in
      let di = CD.get_density_diff_info rd in
      let d, nd = di.di_dens_diff, di.di_dens_ndiff in
      let s =
        match SS.subset d f_diff, SS.subset f_diff d with
        | true, true -> "equal"
        | true, false -> "INEQUAL, comp has fewer elements"
        | false, true -> "INEQUAL, comp has more elements"
        | false, false -> "INEQUAL, incomparable" in
      if flag_fwd then
        begin
          F.printf "%sCOMPARED RESULTS for %S: %s\n" sep_2 name s;
          F.printf "compdiff:  %a\ncompndiff: %a\n" ss_fp d ss_fp nd;
          F.printf "fwddiff:   %a\nfwdndiff:  %a\n" ss_fp f_diff ss_fp f_ndiff;
          F.printf "\n\n%s" sep_1;
        end;
      di
    end
  else
    begin
      F.printf "%s" sep_1;
      { di_dens_diff  = f_diff ;
        di_dens_ndiff = f_ndiff ;
        di_prb_ndiff  = SM.empty ;
        di_val_ndiff  = SM.empty }
    end
