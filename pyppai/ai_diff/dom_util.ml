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
 ** dom_utils.ml: abstract domain utilities *)
open Ir_sig
open Lib


(** Abstract domain for signs and open/closed unit intervals,
 ** i.e. (0,1) and [0,1] *)
module SignDom = struct
  (* OInter = (0,1).
   * CInter = [0,1].
   * Plus   = (0, inf).
   * Minus  = (-inf, 0).
   * Top    = (-inf, inf). *)
  type t = Bot | OInter | CInter | Plus | Minus | Top

  let pp chan s =
    let s_name =
      match s with
      | Bot     -> "Bot"
      | OInter  -> "(0,1)"
      | CInter  -> "[0,1]"
      | Plus    -> "Plus"
      | Minus   -> "Minus"
      | Top     -> "Top" in
    Printf.fprintf chan "%s" s_name
  let fp fmt s =
    let s_name =
      match s with
      | Bot     -> "Bot"
      | OInter  -> "(0,1)"
      | CInter  -> "[0,1]"
      | Plus    -> "Plus"
      | Minus   -> "Minus"
      | Top     -> "Top" in
    F.fprintf fmt "%s" s_name

  let leq s1 s2 =
    match s1, s2 with
    | Bot, _ -> true
    | OInter, (OInter | CInter | Plus | Top) -> true
    | OInter, _ -> false
    | CInter, (CInter | Top) -> true
    | CInter, _ -> false
    | Plus, (Plus | Top) -> true
    | Plus, _ -> false
    | Minus, (Minus | Top) -> true
    | Minus, _ -> false
    | Top, Top -> true
    | Top, _  -> false

  let equal s1 s2 =
    s1 = s2

  let join s1 s2 =
    match s1, s2 with
    | Bot, _ -> s2
    | _, Bot -> s1
    | Top, _ -> s1
    | _, Top -> s2
    | OInter, (OInter | CInter | Plus) -> s2
    | OInter, Minus -> Top
    | CInter, (OInter | CInter) -> s2
    | CInter, (Plus | Minus) -> Top
    | Plus, (OInter | Plus) -> s1
    | Plus, (CInter | Minus) -> Top
    | Minus, Minus -> s1
    | Minus, (OInter | CInter | Plus) -> Top

  let do_uop op s =
    match op, s with
    | _, Bot -> Some Bot
    | Not, _ | SampledStr, _ | SampledStrFmt, _ -> None

  let do_bop op s1 s2 =
    match op, s1, s2 with
    | _, Bot, _ | _, _, Bot -> Some Bot
    | And, _, _ | Or, _, _ -> None
    | Add, (OInter | Plus), (OInter | CInter | Plus) -> Some Plus
    | Add, CInter, (OInter | Plus) -> Some Plus
    | Add, Minus, Minus -> Some Minus
    | Add, _, _ -> Some Top
    | Sub, (OInter | CInter | Plus), Minus -> Some Plus
    | Sub, Minus, (OInter | CInter | Plus) -> Some Minus
    | Sub, _, _ -> Some Top
    | Mult, OInter, OInter -> Some OInter
    | Mult, OInter, CInter -> Some CInter
    | Mult, CInter, (OInter | CInter) -> Some CInter
    | Mult, Plus, (OInter | Plus) -> Some Plus
    | Mult, OInter, Plus -> Some Plus
    | Mult, Minus, Minus -> Some Minus
    | Mult, Minus, (OInter | Plus) -> Some Minus
    | Mult, (OInter | Plus), Minus -> Some Minus
    | Mult, _, _ -> Some Top
    | Div, (OInter | Plus), (OInter | Plus) -> Some Plus
    | Div, Minus, Minus -> Some Plus
    | Div, Minus, (OInter | Plus) -> Some Minus
    | Div, (OInter | Plus), Minus -> Some Minus
    | Div, _, _ -> Some Top
    | Pow, (OInter | Plus), _ -> Some Plus
    | Pow, _, _ -> Some Top

  let do_cop op s1 s2 =
    match s1, s2 with
    | Bot, _
    | _  , Bot -> Some Bot
    | _  , _   -> None

  let do_dist = function
    | Normal              -> Top
    | Exponential         -> Plus
    | Gamma               -> Plus
    | Beta                -> OInter
    | Uniform _           -> Top
    | Dirichlet _         -> OInter
    | Poisson             -> Top
    | Categorical _       -> Top
    | Bernoulli           -> CInter
    | OneHotCategorical _ -> CInter
    | Delta               -> Top
    | Subsample (_, _)    -> Top
    (* new. *)
    | LogNormal           -> Plus
    | ZeroInflatedNegativeBinomial -> Top
    | Multinomial         -> Top
    | VonMises            -> Top
    | MaskedMixtureGammaDelta -> Top
    | MaskedMixtureBetaDelta  -> Top
end

module SD = SignDom
