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
 ** ddom_num.ml: numerical domains for continuity/differentiability analysis *)
open Ir_sig
open Lib

open Apron
open Apron_sig
open Ddom_sig

open Apron_util
open Diff_util
open Dom_util
open Ir_util

(** Numerical domain tracking no information at all
 **   (this is basically the single point { Top } domain) *)
module DN_none =
  (struct
    let name = "top domain"
    let init_domain ~(vars: SS.t): unit = ()
    type t = unit
    let pp _ _ (): unit = ()
    let fp _ _ (): unit = ()
    let is_bot _ = false
    let top () = ()
    let join () () = ()
    let equal () () = true
    let forget _ () = ()
    let assign _ _ () = ()
    let heavoc _ _ () = ()
    let guard _ () = ()
    let call_prim _ _ _ () = ()
    let call_obj _ _ _ () = ()
    let sample _ _ () = ()
    let check_dist_pars _ l () =
      (* maybe change this to true to make no checking at all *)
      Some (List.map (fun _ -> false) l)
    let imply () _ = false
  end: DOM_NUM)

(** Numerical domain based on constants *)
module DN_signs =
  (struct
    let name = "signs"

    (* Set of parameters *)
    let init_domain ~(vars: SS.t): unit = ()

    (* "Variable-Sign":
     * Maps each variable to the sign of the values stored in the
     * variable. Unmapped variables are implicitly mapped to Top.
     * Also, Top includes the possibility of a value being a
     * non-numerical value. When a variable stores a tensor, this
     * map records information about the values stored in the tensor.
     * SD.Top means a numeric value. If not numeric, it should be unbound.
     * *)
    type t = SD.t SM.t

    (* Prtty-printing *)
    let pp (_: string) = ppm SD.pp
    let fp (_: string) = fpm SD.fp

    (* Bottom check: when returns true, definitely bottom *)
    let is_bot _ = false

    (* Lattice operations *)
    let top () = SM.empty
    let join = map_join_inter (fun c0 c1 -> Some (SD.join c0 c1))
    let equal = map_equal (fun v si -> true) SD.equal

    (* Post-condition for assignments *)
    let forget = SM.remove
    let sign_expr (a: t) (e: expr): SD.t option =
      let rec aux = function
        | Nil | True | False | Ellipsis ->
            None
        | Num (Int n) ->
            if n > 0 then Some SD.Plus
            else if n < 0 then Some SD.Minus
            else Some SD.Top
        | Num (Float f) ->
            if f > 0.0 then Some SD.Plus
            else if f < 0.0 then Some SD.Minus
            else Some SD.Top
        | Name x ->
            begin
              try Some (SM.find x a) with Not_found -> None
            end
        | UOp (uop, e0) ->
            bind_opt (aux e0) (fun sg0 ->
              SD.do_uop uop sg0)
        | BOp (bop, e0, e1) ->
            (* bind_opt (aux e0) (fun sg0 ->
             * bind_opt (aux e1) (fun sg1 ->
             *     SD.do_bop bop sg0 sg1)) *)
            (* WL: Added the below adhoc match to prove 1-x>0 for x in (0,1).
             *     This is required to prove dpmm-model. In particular,
             *       beta ~ Beta(...)
             *       beta1m_cumprod = torch.cumprod(1-beta, ...)
             *       mix_weights_beta = F.pad(beta, ...) * F.pad(beta1m_cumprod, ...)
             *       z ~ Categorical(mix_weights_beta)
             *     Here we need to prove mix_weights_beta > 0.*)
            bind_opt (aux e0) (fun sg0 ->
            bind_opt (aux e1) (fun sg1 ->
                match bop, e0, sg1 with
                | Sub, Num (Int n), SD.OInter when n >= 1 -> Some SD.Plus
                | _ -> SD.do_bop bop sg0 sg1))
        | Comp (cop, e0, e1) ->
            bind_opt (aux e0) (fun sg0 ->
            bind_opt (aux e1) (fun sg1 ->
              SD.do_cop cop sg0 sg1))
        | List [] ->
            Some SD.Top
        | List (_ :: _ as el) ->
            let sgl = List.map aux el in
            let f sg_opt_acc sg_opt_cur =
              bind_opt sg_opt_acc (fun sg_acc ->
              bind_opt sg_opt_cur (fun sg_cur ->
                Some (SD.join sg_acc sg_cur))) in
            List.fold_left f (Some SD.Bot) sgl
        | Dict _ | Str _ | StrFmt _ ->
            None in
      aux e
    let assign (x: string) (e: expr) (a: t): t =
      match sign_expr a e with
      | None -> forget x a
      | Some se -> SM.add x se a
    let heavoc (x: string) (c: constr) (a: t): t =
      match c with
      | C_Pos -> SM.add x SD.Plus a
      | C_Neg -> SM.add x SD.Minus a
      | C_Num -> SM.add x SD.Top a


    (* Operation on primitive-function call x=f(el) *)
    let call_prim (x: string) (f: string) (el: expr list) (a: t): t =
      match f, el with
      | "torch.rand", _ ->
          SM.add x SD.Top a
      | "F.softmax", _
      | "F.softplus", _
      | "torch.exp", _ ->
          SM.add x SD.Plus a
      | "torch.ones", _ ->
          SM.add x SD.CInter a
      | "torch.sigmoid", _
      | "TDU.logits_to_probs", _ ->
          SM.add x SD.OInter a
      | "access_with_index", e :: _
      | "int", e :: _
      | "tuple", e :: _
      | "nn.Parameter", e :: _
      | "F.pad", e :: _
      | "torch.FloatTensor", e :: _
      | "torch.LongTensor", e :: _
      | "torch.max", e :: _
      | "torch.reshape", e :: _
      | "torch.squeeze", e :: _
      | "torch.sum", e :: _
      | "torch.tensor", e :: _
      | "torch.Tensor.detach", e :: _
      | "torch.Tensor.long", e :: _
      | "Vindex", e :: _ ->
          begin
            match sign_expr a e with
            | None -> forget x a
            | Some se -> SM.add x se a
          end
      | "torch.log", e :: _ ->
          begin
            match sign_expr a e with
            | Some SD.Plus -> SM.add x SD.Top a
            | Some _
            | None -> forget x a
          end
      | "torch.cumprod", e :: _ ->
          begin
            match sign_expr a e with
            | Some SD.Plus -> SM.add x SD.Plus a
            | _ -> forget x a
          end
      | "torch.matmul", e0 :: e1 :: _ ->
          begin
            match (sign_expr a e0), (sign_expr a e1) with
            | None, _ | _, None ->
                forget x a
            | Some SD.Bot, _ | _, Some SD.Bot ->
                SM.add x SD.Bot a
            | Some SD.Plus, Some SD.Plus | Some SD.Minus, Some SD.Minus ->
                SM.add x SD.Plus a
            | Some SD.Plus, Some SD.Minus | Some SD.Minus, Some SD.Plus ->
                SM.add x SD.Minus a
            | _ ->
                SM.add x SD.Top a
          end
      | "torch.Tensor.view", e :: _ ->
          begin
            match sign_expr a e with
            | None   -> forget x a
            | Some s -> SM.add x s a
          end
      | s, _ ->
          if false then
            F.printf "FORGET(call_prim,signs), function %S ; %s => ?\n" s x;
          forget x a

    (* Operation on an object call x=(c())(el) for an object constructor c *)
    let call_obj (x: string) (c: string) (el: expr list) (a: t): t =
      match c, el with
      | "nn.Sigmoid", _ | "nn.Softmax", _ ->
          SM.add x SD.OInter a
      | "nn.Softplus", _ ->
          SM.add x SD.Plus a
      | "nn.Linear", _ ->
          SM.add x SD.Top a
      | "nn.ReLU", _ -> SM.add x SD.Plus a
      | "nn.BatchNorm1d", _ -> SM.add x SD.Top a
      | s, _ ->
          if false then
            F.printf "FORGET(call_obj,signs), function %S ; %s => ?\n" s x;
          forget x a

    (* Operations on distributions *)
    let sample (x: string) (d: dist) (a: t): t =
      let dist_sign = SD.do_dist (fst d) in
      SM.add x dist_sign a
    let check_dist_pars (d: dist_kind) (el: expr list) (a: t)
        : bool list option =
      let fsig_dom =
        match d with
        | Normal              -> [ Some SD.Top  ; Some SD.Plus ] (* loc, scale *)
        | Exponential         -> [ Some SD.Plus   ]              (* rate *)
        | Gamma               -> [ Some SD.Plus ; Some SD.Plus ] (* concentration, rate *)
        | Beta                -> [ Some SD.Plus ; Some SD.Plus ] (* concentration1, concentration0 *)
        | Uniform _           -> [ Some SD.Bot  ; Some SD.Bot  ] (* low, high *) (* first param < second param *)
        | Dirichlet _         -> [ Some SD.Plus   ] (* concentration *) (* (0,inf)^n for n >= 2 *)
        | Poisson             -> [ Some SD.Plus   ] (* rate  *) (* (0,inf) *)
        | Categorical _       -> [ Some SD.Plus   ] (* probs *) (* (0,inf)^n for n >= 2 *)
        | Bernoulli           -> [ Some SD.CInter ] (* probs *) (* [0,1] *)
        | OneHotCategorical _ -> [ Some SD.Plus   ] (* probs *) (* (0,inf)^n for n >= 2 *)
        | Delta               -> [ None ] (* v *)
        | Subsample (_, _)    -> [ Some SD.Bot  ; Some SD.Bot  ]
        (* new. *)
        | LogNormal     -> [ Some SD.Top  ; Some SD.Plus ] (* loc, scale *)
        | ZeroInflatedNegativeBinomial -> [ Some SD.Plus ; Some SD.Top ; Some SD.Top ] (* total_count, [logits, gate_logits] *)
        | Multinomial   -> [ Some SD.Plus ; Some SD.Plus ] (* total_count, probs *)
        | VonMises      -> [ Some SD.Top  ; Some SD.Top  ] (* loc, concentration *)
        (* mask, gamma_concentration, gamma_rate, delta_value *)
        | MaskedMixtureGammaDelta -> [ Some SD.Top; Some SD.Plus; Some SD.Plus; Some SD.Top ]
        (* mask, beta_concentration1, beta_concentration0, delta_value *)
        | MaskedMixtureBetaDelta  -> [ Some SD.Top; Some SD.Plus; Some SD.Plus; Some SD.Top ]
      in
      let leq sg_opt0 sg_opt1 =
        match sg_opt0, sg_opt1 with
        | _, None -> true
        | None, _ -> false
        | Some sg0, Some sg1 -> SD.leq sg0 sg1 in
      try
        if dbg_distp then F.printf "Sign,check_dist_pars:\n%a" (fp "  ") a;
        let lres = List.map2 leq (List.map (sign_expr a) el) fsig_dom in
        if dbg_distp then
          List.iter2
            (fun e b ->
              F.printf "Sign,check_dist_pars: %a => %b\n" fp_expr e b
            ) el lres;
        Some lres
      with Invalid_argument _ ->
        if dbg_distp then F.printf "Sign,check_dist_pars: invalid\n";
        None

    let imply (a: t) (e_in: expr) : bool =
      let negate_cop op =
        match op with
        | Eq    -> NotEq
        | NotEq -> Eq
        | Lt    -> Gt
        | LtE   -> GtE
        | Gt    -> Lt
        | GtE   -> LtE
        | Is    -> NotIs
        | NotIs -> Is in
      let rec move_not polarity e =
        match e with
        | UOp (Not, e0) ->
            move_not (not polarity) e0
        | UOp _ ->
            if polarity then e else UOp (Not, e)
        | BOp (And, e0, e1) ->
            let e0_neg = move_not polarity e0 in
            let e1_neg = move_not polarity e1 in
            let op_neg = if polarity then And else Or in
            BOp (op_neg, e0_neg, e1_neg)
        | BOp (Or, e0, e1) ->
            let e0_neg = move_not polarity e0 in
            let e1_neg = move_not polarity e1 in
            let op_neg = if polarity then Or else And in
            BOp (op_neg, e0_neg, e1_neg)
        | BOp _ ->
            if polarity then e else UOp (Not, e)
        | Comp (cop, e0, e1) ->
            if polarity then e else Comp (negate_cop cop, e0, e1)
        | _ ->
            if polarity then e else UOp (Not, e) in
      let rec aux e =
        match e with
        | UOp (Not, e0) ->
            begin
              match aux e0 with
              | Some true -> Some false
              | Some false -> Some true
              | None -> None
            end
        | UOp _ ->
            None
        | BOp (And, e0, e1) ->
            begin
              match aux e0, aux e1 with
              | Some true, Some true -> Some true
              | Some false, Some _ | Some _, Some false -> Some false
              | _ -> None
            end
        | BOp (Or, e0, e1) ->
            begin
              match aux e0, aux e1 with
              | Some true, Some _ | Some _, Some true -> Some true
              | Some false, Some false -> Some false
              | _ -> None
            end
        | BOp _ ->
            None
        | Comp (Lt, Num (Float 0.), Num (Float 0.))
        | Comp (Gt, Num (Float 0.), Num (Float 0.)) ->
            Some false
        | Comp (LtE, Num (Float 0.), Num (Float 0.))
        | Comp (GtE, Num (Float 0.), Num (Float 0.)) ->
            Some true
        | Comp (Lt, Num (Float 0.), e0)
        | Comp (Gt, e0, Num (Float 0.)) ->
            begin
               match sign_expr a e0 with
               | None -> None
               | Some SD.Plus -> Some true
               | Some SD.Minus -> Some false
               | _ -> None
            end
        | Comp (Lt, e0, Num (Float 0.))
        | Comp (Gt, Num (Float 0.), e0) ->
            begin
               match sign_expr a e0 with
               | None -> None
               | Some SD.Minus -> Some true
               | Some SD.Plus -> Some false
               | _ -> None
            end
        | Comp (Lt, e0, e1) | Comp (LtE, e0, e1)
        | Comp (Gt, e1, e0) | Comp (GtE, e1, e0) ->
            begin
              match sign_expr a e0, sign_expr a e1 with
              | None, _ | _, None -> None
              | Some SD.Minus, Some SD.Plus -> Some true
              | Some SD.Plus, Some SD.Minus -> Some false
              | _ -> None
            end
        | Comp (Is, Nil, Nil) ->
            Some true
        | Comp (Is, e0, Nil) | Comp (Is, Nil, e0) ->
            begin
              match sign_expr a e0 with
              | Some SD.Plus | Some SD.Minus | Some SD.Top -> Some false
              | _ -> None
            end
        | Comp (NotIs, Nil, Nil) ->
            Some false
        | Comp (NotIs, e0, Nil) | Comp (NotIs, Nil, e0) ->
            begin
              match sign_expr a e0 with
              | Some SD.Plus | Some SD.Minus | Some SD.Top -> Some true
              | _ -> None
            end
        | Comp (NotEq, Num (Float 0.), Num (Float 0.)) ->
            Some false
        | Comp (NotEq, e0, Num (Float 0.))
        | Comp (NotEq, Num (Float 0.), e0) ->
            begin
              match sign_expr a e0 with
              | Some SD.Minus | Some SD.Plus -> Some true
              | _ -> None
            end
        | _ ->
            None in
      match aux (move_not true e_in) with
      | Some true -> true
      | _ -> false

    (* Post-condition for condition test *)
    let guard (e: expr) (a: t): t = a
  end: DOM_NUM)

(** Apron domain generator *)
module DN_apron_make = functor (M: APRON_MGR) ->
  (struct
    let name = F.asprintf "Apron<%s>" M.module_name

    (* Abstraction of the numerical state (variables -> values) *)
    module A = Apron.Abstract1
    let man = M.man
    let a_vars: Apron.Var.t array ref = ref [| |]

    (* Set of parameters *)
    let init_domain ~(vars: SS.t): unit =
      let l = SS.fold List.cons vars [] in
      (*F.printf "init domain: %a\n" ss_fp vars;*)
      a_vars := Array.of_list (List.map make_apron_var l)

    (* Abstract values:
     * - an enviroment
     * - and a conjunction of constraints in Apron representation (u) *)
    type t = M.t A.t

    (* Pretty-printing *)
    let buf_t (ind: string) (buf: Buffer.t) (t: t): unit =
      let lca = A.to_lincons_array man t and nind = ind ^ "  " in
      Printf.bprintf buf "%sApron %d constraints (isbot: %b,vars: %d)\n%a" ind
        (Lincons1.array_length lca) (A.is_bottom man t) (Array.length !a_vars)
        (buf_linconsarray nind) lca
    let pp (ind: string) = buf_to_channel (buf_t ind)
    let fp (ind: string) = buf_to_format (buf_t ind)

    (* Bottom check: when returns true, definitely bottom *)
    let is_bot = A.is_bottom man

    (* Lattice operations *)
    let top (): t =
      let env_empty = Environment.make [| |] !a_vars in
      A.top man env_empty
    let join: t -> t -> t = A.join man
    let equal: t -> t -> bool = A.is_eq man

    (* Post-condition for assignments *)
    let forget (x: string) (t: t): t =
      let var = make_apron_var x in
      A.forget_array man t [| var |] false
    let assign (x: string) (e: expr) (t: t): t =
      if dbg_apron then F.printf "Apron,assign: %a\n" fp_expr e;
      (* convert the expression to Apron IR *)
      try
        let lv = make_apron_var x in
        let rv = Ir_util.make_apron_expr (A.env t) e in
        (* perform the Apron assignment *)
        A.assign_texpr_array man t [| lv |] [| rv |] None
      with e ->
        if dbg_apron then
          F.printf "Apron assign exception, forget (%s)\n"
            (Printexc.to_string e);
        forget x t

    (* Preparation of a condition for Apron *)
    let make_condition (e: expr) (t: t): Tcons1.t =
      if dbg_apron then F.printf "Apron,condition: %a\n" fp_expr e;
      Ir_util.make_apron_cond (A.env t) e

    (* Post-condition for condition test *)
    let guard (e: expr) (t: t): t =
      (*let dbg_apron = true in*)
      if dbg_apron then F.printf "Apron,guard,call: %a\n%a\n" fp_expr e (fp "  ") t;
      try
        let eacons = Tcons1.array_make (A.env t) 1 in
        let cond = make_condition e t in
        if dbg_apron then
          F.printf "Apron,cons: %a\n%a%a" Tcons1.print cond (fp "  ") t A.print t;
        Tcons1.array_set eacons 0 cond;
        let t = A.meet_tcons_array man t eacons in
        (* TODO: bottom reduction *)
        if dbg_apron then
          F.printf "Apron,guard,result %a:\n%a\n%a\n" fp_expr e (fp "  ") t
            A.print t;
        t
      with e ->
        if dbg_apron then
          F.printf "Apron guard exception, ignore (%s)\n"
            (Printexc.to_string e);
        t

    (* Post-condition for heavoc *)
    let heavoc (x: string) (c: constr) (t: t): t =
      match c with
      | C_Pos -> guard (Comp (Gt, Name x, Num (Float 0.))) t
      | C_Neg -> guard (Comp (Lt, Name x, Num (Float 0.))) t
      | C_Num -> forget x t

    (* Satisfiability for condition formula *)
    let sat (e: expr) (t: t): bool =
      if dbg_apron then F.printf "Apron,sat: %a\n" fp_expr e;
      try
        let env = A.env t in
        let ce = Ir_util.make_apron_cond env e in
        A.sat_tcons man t ce
      with e ->
        if dbg_apron then
          F.printf "Apron sat exception, ignore (%s)\n"
            (Printexc.to_string e);
        false

    (* Operation on primitive-function call x=f(el) *)
    let call_prim (x: string) (f: string) (el: expr list) (a: t): t =
      match f with
      | _ -> forget x a

    (* Operation on an object call x=(c())(el) for an object constructor c *)
    let call_obj (x: string) (c: string) (el: expr list) (a: t): t =
      match c with
      | _ -> forget x a

    (* Operations on distributions *)
    let sample (x: string) (d: dist) (t: t): t =
      let t = forget x t in
      let lcons = (* adapted from dom_util.ml:do_dist. *)
        match fst d with
        (* Top *)
        | Normal
        | Uniform None
        | Poisson
        | Categorical _
        | Delta
        | Subsample (_, _)
        | ZeroInflatedNegativeBinomial
        | Multinomial
        | VonMises ->
            [ ]
        (* Plus *)
        | Exponential
        | Gamma
        | LogNormal ->
            [ Comp (Gt, Name x, Num (Float 0.)) ]
        (* CInter *)
        | Bernoulli
        | OneHotCategorical _ ->
            [ Comp (GtE, Name x, Num (Float 0.)) ;
              Comp (LtE, Name x, Num (Float 1.)) ]
        (* OInter *)
        | Beta
        | Dirichlet _ ->
            [ Comp (Gt, Name x, Num (Float 0.)) ;
              Comp (Lt, Name x, Num (Float 1.)) ]
        (* misc *)
        | Uniform (Some (a, b)) ->
            [ Comp (GtE, Name x, Num (Float a)) ;
              Comp (LtE, Name x, Num (Float b)) ]
        | _ -> failwith "ddom_num.ml:sample:lcons." in
      List.fold_left (fun t c -> guard c t) t lcons
    let check_dist_pars (d: dist_kind) (el: expr list) (t: t)
        : bool list option =
      let ollsat =
        let wrong = Comp (LtE, Num (Float 1.), Num (Float 0.)) in
        let wrong_list = List.map (fun _ -> [ wrong ]) el in
        match d, el with
        | Normal, [ e0 ; e1 ] ->
            Some [ [ ] ; [ Comp (Gt, e1, Num (Float 0.)) ] ]
        | Exponential, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Gamma, [ e0 ; e1 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ;
                   [ Comp (Gt, e1, Num (Float 0.)) ] ]
        | Uniform _, [ e0 ; e1 ] ->
            Some [ [ Comp (Lt, e0, e1) ] ;
                   [ Comp (Lt, e0, e1) ] ]
        | Dirichlet _, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Dirichlet _, _ ->
            Some wrong_list
        | Poisson, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Categorical _, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Categorical _, _ ->
            Some wrong_list
        | Bernoulli, _ -> (* false; to check *)
            Some wrong_list
        | OneHotCategorical _, _ -> (* false; to check *)
            Some wrong_list
        | Delta, [ _ ] ->
            Some [ [ ] ]
        | Subsample _, [ e0 ; e1 ] ->
            Some [ [ Comp (GtE, e0, e1) ; Comp (Gt, e0, Num (Float 0.)) ] ;
                   [ Comp (GtE, e0, e1) ; Comp (Gt, e0, Num (Float 0.)) ] ]
        | Subsample (_, _), _ ->
            Some wrong_list
        | _, _ -> None in
      if dbg_distp then F.printf "Apron,check_dist_pars:\n%a" (fp "  ") t;
      let f llsat =
        List.map
          (fun lsat ->
            List.fold_left
              (fun acc e ->
                if dbg_distp then
                  F.printf "Apron,check_dist_pars: %a => %b\n" fp_expr e (sat e t);
                acc && sat e t
              ) true lsat
          ) llsat in
      option_map f ollsat
    let imply (t: t) (e: expr) : bool = sat e t
  end: DOM_NUM)

(** Product domain, with no reduction *)
module DN_prod = functor (D0: DOM_NUM) -> functor (D1: DOM_NUM) ->
  (struct
    let name = F.asprintf "(%s) X (%s)" D0.name D1.name
    let init_domain ~(vars: SS.t): unit =
      D0.init_domain ~vars;
      D1.init_domain ~vars
    type t = D0.t * D1.t
    let pp (ind: string) chan (t0, t1) =
      Printf.fprintf chan "%a%a" (D0.pp ind) t0 (D1.pp ind) t1
    let fp (ind: string) fmt (t0, t1) =
      F.fprintf fmt "%a%a" (D0.fp ind) t0 (D1.fp ind) t1
    let is_bot (t0, t1) = D0.is_bot t0 || D1.is_bot t1
    let top () = D0.top (), D1.top ()
    let join (t0,t1) (u0,u1) = D0.join t0 u0, D1.join t1 u1
    let equal (t0,t1) (u0,u1) = D0.equal t0 u0 && D1.equal t1 u1
    let lift f0 f1 (t0,t1) = f0 t0, f1 t1
    let forget x = lift (D0.forget x) (D1.forget x)
    let assign x e = lift (D0.assign x e) (D1.assign x e)
    let heavoc x c = lift (D0.heavoc x c) (D1.heavoc x c)
    let guard e (t: t): t = lift (D0.guard e) (D1.guard e) t
    let call_prim x f el = lift (D0.call_prim x f el) (D1.call_prim x f el)
    let call_obj x f el = lift (D0.call_obj x f el) (D1.call_obj x f el)
    let sample x d = lift (D0.sample x d) (D1.sample x d)
    let check_dist_pars dk el (t0,t1) =
      match D0.check_dist_pars dk el t0, D1.check_dist_pars dk el t1 with
      | None, None -> None
      | None, Some l | Some l, None -> Some l
      | Some l0, Some l1 -> Some (List.map2 (||) l0 l1)
    let imply (t0,t1) e = D0.imply t0 e || D1.imply t1 e
  end: DOM_NUM)

(** Apron domain instances *)
module DN_box = DN_apron_make( Apron_util.PA_box )
module DN_oct = DN_apron_make( Apron_util.PA_oct )
module DN_pol = DN_apron_make( Apron_util.PA_pol )
(** Apron domain instances, product with signs *)
module DNP_box = DN_prod( DN_signs )( DN_apron_make( Apron_util.PA_box ) )
module DNP_oct = DN_prod( DN_signs )( DN_apron_make( Apron_util.PA_oct ) )
module DNP_pol = DN_prod( DN_signs )( DN_apron_make( Apron_util.PA_pol ) )
