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
 ** ir_util.ml: utilities over the ir form, including pretty-printing *)
open Lib
open Ir_sig
open Ir_ty_sig

open Apron
open Apron_util


(** ***************)
(** string, print *)
(** ***************)
(** Constants into strings *)
let uop_to_string = function
  | Not -> "!"
  | SampledStr -> "sampledS?"
  | SampledStrFmt -> "sampledF?"

let bop_to_string = function
  | And  -> "&&"
  | Or   -> "||"
  | Add  -> "+"
  | Sub  -> "-"
  | Mult -> "*"
  | Div  -> "/"
  | Pow  -> "**"

let cop_to_string = function
  | Eq    -> "=="
  | NotEq -> "!="
  | Is    -> "is"
  | NotIs    -> "notIs"
  | Lt    -> "<"
  | LtE   -> "<="
  | Gt    -> ">"
  | GtE   -> ">="

let dist_kind_to_string_base (dk: dist_kind) =
  match dk with
    (* continuous *)
    | Normal              -> "Normal"
    | Exponential         -> "Exponential"
    | Gamma               -> "Gamma"
    | Beta                -> "Beta"
    | Uniform _           -> "Uniform"
    | Dirichlet _         -> "Dirichlet"
    (* discrete *)
    | Poisson             -> "Poisson"
    | Categorical _       -> "Categorical"
    | Bernoulli           -> "Bernoulli"
    | OneHotCategorical _ -> "OneHotCategorical"
    | Delta               -> "Delta"
    (* subsample *)
    | Subsample _         -> "Subsample"
    (* new *)
    | LogNormal                    -> "LogNormal"
    | ZeroInflatedNegativeBinomial -> "ZeroInflatedNegativeBinomial"
    | Multinomial                  -> "Multinomial"
    | VonMises                     -> "VonMises"
    | MaskedMixtureGammaDelta      -> "MaskedMixtureGammaDelta"
    | MaskedMixtureBetaDelta       -> "MaskedMixtureBetaDelta"

let dist_kind_to_string (dk: dist_kind) =
  let name = dist_kind_to_string_base dk in
  let arg =
    match dk with
    | Uniform arg ->
       begin
         match arg with
         | None -> "?"
         | Some(l,u) ->
            Printf.sprintf "%s,%s"
              (string_of_float l)
              (string_of_float u)
       end
    | Dirichlet         n_case_opt
    | Categorical       n_case_opt
    | OneHotCategorical n_case_opt ->
       int_opt_to_string n_case_opt
    | Subsample (tot, sub) ->
       Printf.sprintf "(%s,%s)"
         (int_opt_to_string tot)
         (int_opt_to_string sub)
    | _ -> "" in
  Printf.sprintf "%s(%s)" name arg

let dist_trans_to_string = function
  | ExpandBy([]) ->
      failwith "dist_trans_to_string: invariant violation"
  | ExpandBy(x::xs) ->
      let x_str = int_opt_to_string x in
      let arg_str = List.fold_left (fun acc xi -> acc ^ ", " ^ (int_opt_to_string xi)) x_str xs in
      "expand_by([" ^ arg_str ^ "])"
  | ToEvent(x) -> "to_event(" ^ (int_opt_to_string x) ^ ")"

let dist_trans_list_to_string l =
  List.fold_left (fun acc dist_trans -> acc ^ "." ^ (dist_trans_to_string dist_trans)) "" l

let dist_to_string (dist_kind, dist_trans_l) =
  let dist_kind_str = dist_kind_to_string dist_kind in
  let dist_trans_l_str = dist_trans_list_to_string dist_trans_l in
  dist_kind_str ^ dist_trans_l_str

(** Functions to print into buffers *)
let buf_dist buf d =
  Printf.bprintf buf "%s" (dist_to_string d)

let buf_number buf = function
  | Int i -> Printf.bprintf buf "%d" i
  | Float f -> Printf.bprintf buf "%f" f

let rec buf_lift_to_list pp buf = function
  | [] -> ()
  | [x] -> pp buf x
  | x::l -> Printf.bprintf buf "%a, %a" pp x (buf_lift_to_list pp) l


let rec buf_expr buf = function
  | Nil -> Printf.bprintf buf "nil"
  | True -> Printf.bprintf buf "true"
  | False -> Printf.bprintf buf "false"
  | Ellipsis -> Printf.bprintf buf "..."
  | Num n -> buf_number buf n
  | Name n -> Printf.bprintf buf "%s" n
  | UOp (u, e) -> Printf.bprintf buf "(%s %a)" (uop_to_string u) buf_expr e
  | BOp (b, e0, e1) ->
      Printf.bprintf buf "(%a %s %a)" buf_expr e0 (bop_to_string b) buf_expr e1
  | Comp (c, e0, e1) ->
      Printf.bprintf buf "(%a %s %a)" buf_expr e0 (cop_to_string c) buf_expr e1
  | List (es) -> Printf.bprintf buf "[%a]" buf_expr_list es
  | Dict (ks, vs) -> Printf.bprintf buf "{[%a]->[%a]}" buf_expr_list ks buf_expr_list vs
  | Str s -> Printf.bprintf buf "\"%s\"" s
  | StrFmt (s, l) -> Printf.bprintf buf "\"%s\".format(%a)" s buf_expr_list l

and buf_expr_list l = buf_lift_to_list buf_expr l

let buf_keyword buf = function
  | (None, e) -> Printf.bprintf buf "_ -> %a" buf_expr e
  | (Some id, e) -> Printf.bprintf buf "%s -> %a" id buf_expr e

let buf_keyword_list = buf_lift_to_list buf_keyword

let buf_acmd buf = function
  | Assert e -> Printf.bprintf buf "assert(%a)" buf_expr e
  | Assume e -> Printf.bprintf buf "assume(%a)" buf_expr e
  | Assn (n, e) -> Printf.bprintf buf "%s = %a" n buf_expr e
  | AssnCall (n, e, l, kw_l) ->
      Printf.bprintf buf "%s = %a([%a],[%a])" n buf_expr e buf_expr_list l
        buf_keyword_list kw_l
  | Sample (n, e0, d, l, obs_opt, repar) ->
      Printf.bprintf buf "sample(%s, %a, %s, [%a]" n buf_expr e0
        (dist_to_string d) buf_expr_list l;
      begin
        match obs_opt with
        | None -> ()
        | Some e1 -> Printf.bprintf buf ", obs=%a" buf_expr e1
      end;
      if repar then Printf.bprintf buf ", repar";
      Printf.bprintf buf ")"

let buf_withitem buf = function
  | (e1, None) ->
      Printf.bprintf buf "%a" buf_expr e1
  | (e1, Some e2) ->
      Printf.bprintf buf "%a as %a" buf_expr e1 buf_expr e2

let buf_withitem_list = buf_lift_to_list buf_withitem

let rec buf_stmt_ind ind buf = function
  | Atomic a -> Printf.bprintf buf "%s%a;\n" ind buf_acmd a
  | If (e, pt, pf) ->
      let nind = "  "^ind in
      Printf.bprintf buf "%sif(%a)\n%a%selse\n%a" ind buf_expr e
        (buf_block_ind nind) pt ind (buf_block_ind nind) pf
  | For (ei, er, p) ->
      let nind = "  "^ind in
      Printf.bprintf buf "%sfor(%a in %a)\n%a" ind buf_expr ei buf_expr er
        (buf_block_ind nind) p
  | While (e, p) ->
      let nind = "  "^ind in
      Printf.bprintf buf "%swhile(%a)\n%a" ind buf_expr e
        (buf_block_ind nind) p
  | Break -> Printf.bprintf buf "%sbreak;\n" ind
  | Continue -> Printf.bprintf buf "%scontinue;\n" ind
  | With(item_list, p) ->
      let nind = "  "^ind in
      Printf.bprintf buf "%swith[%a]\n%a" ind
                     buf_withitem_list item_list
                     (buf_block_ind nind) p
and buf_block_ind ind buf =
  List.iter (buf_stmt_ind ind buf)

let buf_stmt  = buf_stmt_ind ""
let buf_block = buf_block_ind ""
let buf_prog  = buf_block_ind ""

(** Conversion to strings *)
let number_to_string = buf_to_string buf_number
let expr_to_string   = buf_to_string buf_expr
let acmd_to_string   = buf_to_string buf_acmd
let prog_to_string   = buf_to_string buf_prog

(** Pretty-printing on channels *)
let pp_number    = buf_to_channel buf_number
let pp_expr      = buf_to_channel buf_expr
let pp_expr_list = buf_to_channel buf_expr_list
let pp_acmd      = buf_to_channel buf_acmd
let pp_stmt      = buf_to_channel buf_stmt
let pp_block     = buf_to_channel buf_block
let pp_prog      = buf_to_channel buf_prog

(** Pretty-printing on formatters *)
let fp_dist      = buf_to_format buf_dist
let fp_number    = buf_to_format buf_number
let fp_expr      = buf_to_format buf_expr
let fp_expr_list = buf_to_format buf_expr_list
let fp_acmd      = buf_to_format buf_acmd
let fp_stmt      = buf_to_format buf_stmt
let fp_block     = buf_to_format buf_block
let fp_prog      = buf_to_format buf_prog
let fpi_block ind = buf_to_format (buf_block_ind ind)


(** ********************)
(** functions for expr *)
(** ********************)
(** to_{int,string}_opt *)
let expr_to_int_opt = function
  | Num(Int n) -> Some n
  | _ -> None

let expr_to_string_opt = function
  | Str(str) -> Some str
  | _ -> None

(** Functions for simplifying expressions *)
(* negate_cop negates the comparison operator *)
let negate_cop = function
  | Eq -> NotEq
  | NotEq -> Eq
  | Is -> NotIs
  | NotIs -> Is
  | Lt -> GtE
  | LtE -> Gt
  | Gt -> LtE
  | GtE -> Lt

let incr_expr = function
  | Num (Int n) -> Num(Int (n + 1))
  | e           -> BOp(Add, e, Num(Int(1)))

let decr_expr = function
  | Num (Int n) -> Num(Int (n - 1))
  | e           -> BOp(Sub, e, Num(Int(1)))

(* negate_expr negates an expression if the expression is a boolean expression
 * and performs a form of partial evaluation. It follows the semantics of
 * python's not operator. The argument is a function that performs an (approximate)
 * type inference. *)
let rec negate_exp infer_ty = function
  | Nil ->
      True
  | True ->
      False
  | False ->
      True
  | Ellipsis ->
      False
  | Num (Int(n)) ->
      if (n = 0) then True else False
  | Num (Float(f)) ->
      if (f = 0.0) then True else False
  | Name _ as e ->
      UOp(Not, e)
  | UOp(uop, e0) as e ->
      begin
        match uop with
        | Not -> e0
        | SampledStr | SampledStrFmt -> UOp(Not, e)
      end
  | BOp(bop, e0, e1) as e ->
      begin
        match bop with
        | And -> BOp(Or, negate_exp infer_ty e0, negate_exp infer_ty e1)
        | Or -> BOp(And, negate_exp infer_ty e0, negate_exp infer_ty e1)
        | Add | Sub | Mult | Div | Pow -> UOp(Not, e)
      end
  | Comp(cop, e0, e1) ->
      begin
        let e0_is_int = infer_ty e0 = ET_num (NT_int) in
        let e1_is_int = infer_ty e1 = ET_num (NT_int) in
        match cop with
        | LtE when e0_is_int && e1_is_int ->
            Comp(GtE, e0, incr_expr e1)
        | GtE when e0_is_int && e1_is_int ->
            Comp(LtE, e0, decr_expr e1)
        | _ -> Comp(negate_cop cop, e0, e1)
      end
  | List(es) ->
      if (es = []) then True else False
  | Dict(ks, vs) ->
      if (ks = []) then True else False
  | Str(s) ->
      if (s = "") then True else False
  | StrFmt _ as e ->
      UOp(Not, e)

let rec simplify_exp infer_ty e =
  match e with
  | Nil | True | False | Ellipsis | Num _ ->
      e
  | Name _ ->
      if infer_ty e = ET_nil then Nil else e
  | UOp(Not, e0) ->
      let e0_new = simplify_exp infer_ty e0 in
      negate_exp infer_ty e0_new
  | UOp(uop, e0) ->
      let e0_new = simplify_exp infer_ty e0 in
      if e0_new == e0 then e
      else UOp(uop, e0_new)
  | BOp(bop, e0, e1) as e ->
      let e0_new = simplify_exp infer_ty e0 in
      let e1_new = simplify_exp infer_ty e1 in
      let default_e =
        if (e0_new == e0 && e1_new == e1) then e
        else BOp(bop, e0_new, e1_new) in
      begin
        match bop with
        | And ->
            (match e0_new, e1_new with
             | False, _ | _, False -> False
             | True, _ -> e1_new
             | _, True -> e0_new
             | _ -> default_e)
        | Or ->
            (match e0_new, e1_new with
             | False, _ -> e1_new
             | _, False -> e0_new
             | True, _ | _, True  -> True
             | _ -> default_e)
        | Add ->
            (match e0_new, e1_new with
             | Num (Int n0), Num (Int n1) -> Num (Int (n0 + n1))
             | Num (Float f0), Num (Float f1) -> Num (Float (f0 +. f1))
             | _ -> default_e)
        | Sub ->
            (match e0_new, e1_new with
             | Num (Int n0), Num (Int n1) -> Num (Int (n0 - n1))
             | Num (Float f0), Num (Float f1) -> Num (Float (f0 -. f1))
             | _ -> default_e)
        | Mult ->
            (match e0_new, e1_new with
             | Num (Int n0), Num (Int n1) -> Num (Int (n0 * n1))
             | Num (Float f0), Num (Float f1) -> Num (Float (f0 *. f1))
             | _ -> default_e)
        | Div ->
            (match e0_new, e1_new with
             (* int / int can be float. e.g., 1 / 3 = 0.33... *)
             | Num (Int n0), Num (Int n1) ->
                Num (Float (float_of_int(n0) /. float_of_int(n1)))
             | Num (Float f0), Num (Float f1) -> Num (Float (f0 /. f1))
             | _ -> default_e)
        | Pow ->
            (match e0_new, e1_new with
             (* int ** int can be float. e.g., 2 ** -1 = 0.5 *)
             | Num (Int n0), Num (Int n1) ->
                Num (Float (float_of_int(n0) ** float_of_int(n1)))
             | Num (Float f0), Num (Float f1) -> Num (Float (f0 ** f1))
             | _ -> default_e)
      end
  | Comp(cop, e0, e1) ->
      let e0_new = simplify_exp infer_ty e0 in
      let e1_new = simplify_exp infer_ty e1 in
      let e0_ty = infer_ty e0_new in
      let e1_ty = infer_ty e1_new in
      let default_e =
        if (e0_new == e0 && e1_new == e1) then e
        else Comp(cop, e0_new, e1_new) in
      begin
        match cop with
        | Lt when e0_ty = ET_num (NT_int) && e1_ty = ET_num (NT_int) ->
            Comp(LtE, e0_new, decr_expr e1_new)
        | Gt when e0_ty = ET_num (NT_int) && e1_ty = ET_num (NT_int) ->
            Comp(GtE, e0_new, incr_expr e1_new)
        | Is | Eq ->
            if e0_ty = ET_nil && e1_ty = ET_nil then
              True
            else if e0_ty <> ET_unknown && e1_ty <> ET_unknown && e0_ty <> e1_ty then
              False
            else
              default_e
        | NotIs | NotEq ->
            if e0_ty = ET_nil && e1_ty = ET_nil then
              False
            else if e0_ty <> ET_unknown && e1_ty <> ET_unknown && e0_ty <> e1_ty then
              True
            else
              default_e
        | _ ->
            default_e
      end
  | List _ | Dict _ | Str _ | StrFmt _ ->
      e


(** ********************)
(** functions for dist *)
(** ********************)
let dist_kind_support_subseteq dk1 dk2 =
  match dk2 with
  (* continuous *)
  | Normal ->
      (match dk1 with
        | Normal | Exponential | Gamma | Beta | Uniform _ -> true
        | _ -> false)
  | Exponential ->
      (match dk1 with
        | Exponential | Gamma | Beta -> true
        | Uniform (Some (l,u)) -> l >= 0.0
        | _ -> false)
  | Gamma ->
      (match dk1 with
        | Gamma | Beta -> true
        | Uniform (Some (l,u)) -> l > 0.0
        | _ -> false)
  | Beta ->
      (match dk1 with
        | Beta -> true
        | Uniform (Some (l,u)) -> 0.0 < l && u < 1.0
        | _ -> false)
  | Uniform bound_opt2 ->
      (match bound_opt2, dk1 with
        | Some (l2,u2), Beta -> l2 <= 0.0 && 1.0 <= u2
        | Some (l2,u2), Uniform (Some (l1, u1)) -> l2 <= l1 && u1 <= u2
        | _ -> false)
  | Dirichlet n_case_opt2 ->
      (match (dk1, n_case_opt2) with
        | Dirichlet (Some n_case1), Some n_case2 -> n_case1 = n_case2
        | _ -> false)
  (* discrete *)
  | Poisson ->
      (match dk1 with
        | Poisson | Categorical _ | Bernoulli -> true
        | _ -> false)
  | Categorical n_case_opt2 ->
      (match dk1 with
        | Categorical n_case_opt1 -> opt_some_leq n_case_opt1 n_case_opt2
        | Bernoulli -> true
        | _ -> false)
  | Bernoulli ->
      (match dk1 with
        | Bernoulli -> true
        | _ -> false)
  | OneHotCategorical n_case_opt2 ->
      (match (dk1, n_case_opt2) with
        | OneHotCategorical (Some n_case1), Some n_case2 -> n_case1 = n_case2
        | _ -> false)
  | Delta ->
      false (* wy: very imprecise. can be improved. *)
  (* subsample *)
  | Subsample (size_opt2, subsize_opt2) ->
      (match (size_opt2, subsize_opt2, dk1) with
        | Some t2, Some s2, Subsample (Some t1, Some s1) -> s1 = s2 && t1 <= t2
        | _ -> false)
  (* new *)
  | _ ->
     Format.printf "TODO: dist_kind_support_subseteq: unimplemented case (%s,%s)"
       (dist_kind_to_string dk1)
       (dist_kind_to_string dk2);
     false


(** ********************)
(** functions for stmt *)
(** ********************)
(** Checking various properties of statements *)
let rec contains_continue: stmt -> bool = function
  | Atomic _ -> false
  | If (_, body, orelse) ->
     (List.exists contains_continue body) || (List.exists contains_continue orelse)
  | For (_, _, body) | While (_, body) | With (_, body) ->
     (List.exists contains_continue body)
  | Break    -> false
  | Continue -> true

let rec contains_break: stmt -> bool = function
  | Atomic _ -> false
  | If (_, body, orelse) ->
     (List.exists contains_break body) || (List.exists contains_break orelse)
  | For (_, _, body) | While (_, body) | With (_, body) ->
     (List.exists contains_break body)
  | Break    -> true
  | Continue -> false

(*
(* wy: currently `contains_return' always returns false,
 *     because our IR has no `Return' constructor.
 *     maybe need to consider adding `Return' to ir_sig,
 *     because the translation of `Return' of pyast to ir seems wrong;
 *     see `_stmt_to_prog', case `Pya.Return', in ir_cast.ml;
 *     e.g., `return 1; return 2' will be translated to
 *     `__@@ret=1; __@@ret=2' which is wrong. *)
let rec contains_return: stmt -> bool = function
  | Atomic _ -> false
  | If (_, body, orelse) ->
     (List.exists contains_return body) || (List.exists contains_return orelse)
  | For (_, _, body) | While (_, body) | With (_, body) ->
     (List.exists contains_return body)
  | Break | Continue -> false
 *)


(** ********************)
(** functions for prog *)
(** ********************)
(* Simplify use of Delta distributions in `sample` (not in `observe`) as follows:
 * replace each `Sample(trgt, _, (Delta, []), [arg], None)` by `Assn(trgt, arg)`. *)
let _simpd_acmd: acmd -> acmd = function
  | Sample(trgt, name, d, args, obsv, _) as acmd ->
     begin
       match d, args, obsv with
       | (Delta, []), [arg], None -> Assn(trgt, arg)
       | _ -> acmd
     end
  | acmd -> acmd

let rec _simpd_stmt: stmt -> stmt = function
  | Atomic(acmd) ->
      Atomic(_simpd_acmd acmd)
  | If(cond, body, orelse) ->
      If(cond, _simpd_block body, _simpd_block orelse)
  | For(i, range, body) ->
      For(i, range, _simpd_block body)
  | While(cond, body) ->
      While(cond, _simpd_block body)
  | With(items, body) ->
      With(items, _simpd_block body)
  | (Break | Continue) as stmt ->
      stmt

and _simpd_block (block: block): block =
  List.map _simpd_stmt block

let simplify_delta_prog = _simpd_block


(** ************************)
(** Apron helper functions *)
(** ************************)
(* Conversion of an IR expr into an Apron expression
 * (this function is very conservative and rejects many expressions) *)
let make_apron_expr (env: Apron.Environment.t): expr -> Texpr1.t =
  let rec aux (e: expr): Texpr1.t =
    let fail () =
      failwith (Printf.sprintf "unhandled expr: %S" (expr_to_string e)) in
    match e with
    | True | False | Str _ | StrFmt _ ->
        fail ()
    | Num (Int i) -> Texpr1.cst env (Coeff.s_of_int i)
    | Num (Float f) -> Texpr1.cst env (Coeff.s_of_float f)
    | Name id -> Texpr1.var env (make_apron_var id)
    | BOp (b, e0, e1) ->
        let b =
          match b with
          | Add -> Texpr0.Add
          | Sub -> Texpr0.Sub
          | Mult -> Texpr0.Mul
          | Div -> Texpr0.Div
          | _ -> failwith "binary operator" in
        Texpr1.binop b (aux e0) (aux e1) Texpr1.Real Texpr1.Near
    | _ -> fail () in
  aux
let make_apron_cond (env: Apron.Environment.t) (e: expr)
    : Tcons1.t =
  match simplify_exp (fun _ -> ET_unknown) e with
  | True -> (* tautology constraint 0 = 0 *)
      let ex = make_apron_expr env (Num (Int 0)) in
      Tcons1.make ex Tcons1.EQ
  | False -> (* anti-tautology constraint 1 = 0 *)
      let ex = make_apron_expr env (Num (Int 1)) in
      Tcons1.make ex Tcons1.EQ
  | Comp (c, e0, e1) ->
      (*   e0 (c) e1    is translated into    e0 - e1 (c) 0   *)
      let c, e0, e1 =
        match c with
        | Eq    -> Lincons1.EQ   , e0, e1
        | NotEq -> Lincons1.DISEQ, e0, e1
        | Is    -> Lincons1.EQ   , e0, e1
        | NotIs -> Lincons1.DISEQ, e0, e1
        | Lt    -> Lincons1.SUP  , e1, e0
        | LtE   -> Lincons1.SUPEQ, e1, e0
        | Gt    -> Lincons1.SUP  , e0, e1
        | GtE   -> Lincons1.SUPEQ, e0, e1 in
      let ex = make_apron_expr env (BOp (Sub, e0, e1)) in
      Tcons1.make ex c
  | _ ->
      (* todo *)
      let ex = make_apron_expr env (Num (Int 0)) in
      Tcons1.make ex Tcons1.EQ
        (* failwith (Printf.sprintf "make_apron_cond: %s"
           (expr_to_string e)) *)
(* Make thresholds to use for widening *)
(* xr: this is a bit experimental now so I consider only one
 * expression for now, but we can generalize later *)
let make_thr env (el: expr list): Lincons1.earray =
  match el with
  | [ e ] ->
      begin
        match e with
        | Comp (Lt, Name x, Name y) ->
            let ea = Lincons1.array_make env 2 in
            let le = Linexpr1.make env in
            Linexpr1.set_list le
              [ Coeff.s_of_int (-1), make_apron_var x ;
                Coeff.s_of_int   1 , make_apron_var y ]
              (Some (Coeff.s_of_int 0));
            (* threshold -x + y >  0 *)
            let lcgt = Lincons1.make le Lincons1.SUP
            (* threshold -x + y >= 0 *)
            and lcge = Lincons1.make le Lincons1.SUPEQ in
            Lincons1.array_set ea 0 lcgt;
            Lincons1.array_set ea 1 lcge;
            ea
        | _ ->
            failwith (Printf.sprintf "todo:make_thr:%s" (expr_to_string e))
      end
  | _ -> failwith "several expresions"


(** ***************************)
(** Analysis helper functions *)
(** ***************************)
(* Helper functions: extract range info (int^3 opt) from args of range of plate.
 * Function list:
 * - _range_info_from_{range,plate} *)
let range_info_from_range: int option list -> (int*int*int) option = function
  | [Some u]                 -> Some (0, u, 1)
  | [Some l; Some u]         -> Some (l, u, 1)
  | [Some l; Some u; Some s] -> Some (l, u, s)
  | _                        -> None

(* Extraction of range from plate *)
let range_info_from_plate: int option list -> (int*int*int) option = function
  | [] ->
      failwith "_range_info_from_plate:todo:1" (* wy: not obvious... *)
  | [Some size] ->
      Some (0, size, 1)
  | [Some size; Some subsize] when size = subsize ->
      Some (0, size, 1)
  | [Some size; _] ->
      failwith "_range_info_from_plate:todo:2" (* wy: not obvious... *)
  | _ ->
      failwith "_range_info_from_plate:error:1"

(* Range info for a loop *)
let range_info (et: exp_ty): (int * int * int) option =
  match et with
  | ET_plate (Some (pt_s_args, _)) -> range_info_from_plate pt_s_args
  | ET_range (Some rt_args)        -> range_info_from_range rt_args
  | _ -> None

(* Init, condition, increment for a for loop *)
let cond_of_for_loop (i: idtf) (range: (int * int * int) option)
    : stmt * expr * stmt =
  let (cmp, e_l, e_u, e_s): cop * expr * expr * expr =
    match range with
    | None -> failwith "eval_stmt: for: unimplemented case 1"
    | Some (l, u, s) ->
        (if      s > 0 then Lt
        else if s < 0 then Gt
        else failwith "eval_stmt: for: error 1"),
        Num (Int l), Num (Int u), Num (Int s) in
  (* init_i= i=e_l *** FINAL
   * inc_i = i=i+e_s
   * test_i= i<e_u *)
  let init_i = Atomic (Assn (i, e_l)) in
  let inc_i  = Atomic (Assn (i, BOp (Add, Name i, e_s))) in
  let test_i = Comp(cmp, Name i, e_u) in
  init_i, test_i, inc_i
