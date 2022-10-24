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
 ** diff_util.ml: utilities for the differentiability analysis *)
open Lib

open Ir_sig

open Ir_util


(** Some debugging flags *)
let dbg_apron   = false
let dbg_init    = false
let dbg_join    = false
let dbg_compose = false
let dbg_call    = false
let dbg_param   = false
let dbg_module  = false
let dbg_sample  = false
let dbg_observe = false
let dbg_distp   = false
let dbg_loop    = false
let dbg_loc     = false (* for localisation *)

(** Differentiability-related properties. *)
(* type for properties related to differentiability:
 *   Diff: differentiable w.r.t. some parameters S.
 *   Lips: locally Lipschitz w.r.t. some parameters S.
 *   Top:  always true. *)
type diff_prop = Diff | Lips | Top

(** Utilities for the printing of results *)
let fp_diff_prop_short fmt = function
  | Diff -> F.fprintf fmt "diff."
  | Lips -> F.fprintf fmt "lipsch. cont."
  | Top  -> F.fprintf fmt "_"
let fp_diff_prop fmt = function
  | Diff -> F.fprintf fmt "differentiability"
  | Lips -> F.fprintf fmt "Lipschitz continuity"
  | Top  -> F.fprintf fmt "trivial smoothness abstraction {Top}"
let fp_for_diff_prop fmt = function
  | Diff -> F.fprintf fmt "for differentiability"
  | Lips -> F.fprintf fmt "for Lipschitz continuity"
  | Top  -> F.fprintf fmt "for trivial smoothness property (smoothness abstraction off)"
let ppm pp chan m =
  SM.iter (fun s -> Printf.fprintf chan "\t%-30s\t=>\t%a\n" s pp) m
let fpm fp fmt m =
  SM.iter (fun s -> F.fprintf fmt "\t%-30s\t=>\t%a\n" s fp) m
let fp_dist_par fmt = function
  | None -> F.fprintf fmt "<None>"
  | Some l ->
      F.fprintf fmt "[";
      List.iter (F.fprintf fmt "%b;") l;
      F.fprintf fmt "]"

(** Utilities for handling options *)
let bind_opt x f =
  match x with
  | None -> None
  | Some x0 -> f x0

(** Utilities for the operations in the diff domain *)
let map_join_union join m0 m1 =
  SM.fold
    (fun v0 c0 acc ->
      try SM.add v0 (join c0 (SM.find v0 m1)) acc
      with Not_found -> SM.add v0 c0 acc
    ) m0 m1
let map_join_inter join m0 m1 =
  SM.fold
    (fun v0 c0 acc ->
      try match join c0 (SM.find v0 m1) with
      | None -> acc
      | Some c -> SM.add v0 c acc
      with Not_found -> acc
    ) m0 SM.empty
let map_equal pred eq m0 m1 =
  let m0 = SM.filter pred m0 in
  let m1 = SM.filter pred m1 in
  let ck v c0 = try eq c0 (SM.find v m1) with Not_found -> false in
  SM.cardinal m0 = SM.cardinal m1 && SM.for_all ck m0
let lookup_with_default (k: string) (m: 'a SM.t) (default: 'a) : 'a =
  try SM.find k m with Not_found -> default

(** Preanalysis to get the set of parameters for which the differentiability
 ** analysis should provide results *)
let prog_varparams ~(getpars: bool) ~(getvars: bool) (p: prog): SS.t =
  let aux_id acc id =
    if getvars then SS.add id acc
    else acc in
  let aux_par acc id =
    if getpars then SS.add id acc
    else acc in
  let rec aux_expr (acc: SS.t): expr -> SS.t = function
    | Nil | True | False | Num _ | Str _ | Ellipsis -> acc
    | Name id -> aux_id acc id
    | UOp (_, e) -> aux_expr acc e
    | BOp (_, e0, e1) | Comp (_, e0, e1) -> aux_expr (aux_expr acc e0) e1
    | List el | StrFmt (_, el) -> aux_expr_list acc el
    | Dict (el0, el1) -> aux_expr_list (aux_expr_list acc el0) el1
  and aux_expr_list (acc: SS.t): expr list -> SS.t =
    List.fold_left aux_expr acc in
  let aux_acmd (acc: SS.t): acmd -> SS.t = fun acmd ->
    match acmd with
    | Assert e | Assume e -> aux_expr acc e
    | Assn (id, e) -> aux_expr (aux_id acc id) e
    | AssnCall (x, Name "pyro.param", Str pname :: Name _ :: _, _)
    | AssnCall (x, Name "pyro.param", Str pname :: _, _)
    | AssnCall (x, Name "pyro.module", Str pname :: Name _ :: _, _ ) ->
        aux_id (aux_par acc pname) x
    | Sample (x, Str pname, _, _, eo, _) ->
        (* this case cannot filterout the no_obs case as in ai_diff... *)
        (* TODO: add variables in arguments *)
        let acc =
          match eo with
          | None -> acc
          | Some e -> aux_expr acc e in
        aux_id (aux_par acc pname) x
    | AssnCall (x, Name _, el, _) ->
        aux_expr_list (aux_id acc x) el
    | AssnCall (_, _, _, _) -> failwith "unhandled case"
    | Sample (_, e, _, el, eo, _) ->
        (* TODO: add variables in arguments *)
        let acc =
          match eo with
          | None -> acc
          | Some e -> aux_expr acc e in
        acc in
  let rec aux_stmt (acc: SS.t): stmt -> SS.t = function
    | Atomic ac -> aux_acmd acc ac
    | If (e, b0, b1) -> aux_block (aux_block (aux_expr acc e) b0) b1
    | For (e0, e1, b) -> aux_block (aux_expr (aux_expr acc e0) e1) b
    | While (e, b) -> aux_block (aux_expr acc e) b
    | With (l, b) ->
        let acc =
          List.fold_left
            (fun acc (e, o) ->
              let acc = aux_expr acc e in
              match o with
              | None -> acc
              | Some e -> aux_expr acc e
            ) acc l in
        aux_block acc b
    | Break | Continue -> acc
  and aux_block (acc: SS.t): block -> SS.t = List.fold_left aux_stmt acc in
  aux_block SS.empty p


(** Abstraction of dependences in an expression *)
let dep_expr (e: expr): SS.t =
  let rec aux_expr acc = function
    | Nil | True | False | Ellipsis | Num _ | Str _ -> acc
    | Name x -> SS.add x acc
    | BOp (_, e0, e1) | Comp (_, e0, e1) -> aux_expr (aux_expr acc e0) e1
    | List el | StrFmt (_, el) -> List.fold_left aux_expr acc el
    | e ->
        F.printf "TODO[dep_expr]:\n  %a\n\n\n" fp_expr e; flush stdout;
        failwith "dep_expr" in
  aux_expr SS.empty e

(** A very basic, hackish localisation setup *)
module Loc =
  struct
    type u = Block | Ift | Iff | Loop | Par of int
    type t = (u * int) list
    let fp fmt (t: t) =
      let u_to_string = function
        | Block -> "bl"
        | Ift   -> "if:t"
        | Iff   -> "if:f"
        | Loop  -> "loop"
        | Par i -> F.asprintf "p:%d" i in
      let b = ref false in
      F.fprintf fmt "{";
      List.iter
        (fun (u,i) ->
          F.fprintf fmt "%s%s@%d" (if !b then ";" else "") (u_to_string u) i;
          b := true
        ) t;
      F.fprintf fmt "}"
    let compare = Stdlib.compare
    let start = [ Block, 0 ]
    let next = function
      | [ ] -> failwith "empty localization"
      | (u, n) :: t -> (u, n+1) :: t
    let if_t t = (Ift, 0) :: t
    let if_f t = (Iff, 0) :: t
    let loop t = (Loop, 0) :: t
    let parn t n = (Par n, 0) :: t
  end
module SLoc = Set.Make( Loc )
module MLoc = Map.Make( Loc )

(** A basic domain to track safety information *)
module SI =
  struct
    let dbg_loc = false
    (* None: top, no information
     * Some s: elements of s are unsafe *)
    type t =
        { (* whether absence of division by 0 could be proved *)
          si_no_div0: SLoc.t option ;
          (* whether all distribution parameters are proved ok *)
          si_pars_ok: SLoc.t option }
    (* General information *)
    let nowhere_m mo =
      match mo with
      | None -> false
      | Some s -> s = SLoc.empty
    let nowhere_div0 t = nowhere_m t.si_no_div0
    let nowhere_par_ko t = nowhere_m t.si_pars_ok
    (* Printing *)
    let pp chan t =
      let si_no_div0 = nowhere_div0 t
      and si_pars_ok = nowhere_par_ko t in
      Printf.fprintf chan "(%s,%s)"
        (if si_no_div0 then "no div0" else "div0:?")
        (if si_pars_ok then "pars ok" else "pars:?")
    let fp fmt t =
      let si_no_div0 = nowhere_div0 t
      and si_pars_ok = nowhere_par_ko t in
      F.fprintf fmt "(%s,%s)"
        (if si_no_div0 then "no div0" else "div0:?")
        (if si_pars_ok then "pars ok" else "pars:?")
    (* No information *)
    let top = { si_no_div0 = None ;
                si_pars_ok = None }
    (* Information for identity transformer *)
    let id  = { si_no_div0 = Some SLoc.empty ;
                si_pars_ok = Some SLoc.empty }
    (* Lattice operations *)
    let join (t0: t) (t1: t): t =
      let f o0 o1 =
        match o0, o1 with
        | None, _ | _, None -> None
        | Some s0, Some s1 -> Some (SLoc.union s0 s1) in
      { si_no_div0 = f t0.si_no_div0 t1.si_no_div0 ;
        si_pars_ok = f t0.si_pars_ok t1.si_pars_ok }
    let equal (t0: t) (t1: t): bool = true
    (* Get information about safety *)
    let no_div0 (t: t) (loc: Loc.t): bool =
      match t.si_no_div0 with
      | None -> false
      | Some s ->
          let b = not (SLoc.mem loc s) in
          if dbg_loc then F.printf "Loc:Ask:Div at %a: %b\n" Loc.fp loc b;
          b
    let par_ok (t: t) (loc: Loc.t): bool =
      match t.si_pars_ok with
      | None -> false
      | Some s ->
          let b = not (SLoc.mem loc s) in
          if dbg_loc then F.printf "Loc:Ask:Par at %a: %b\n" Loc.fp loc b;
          b
    (* Accumulate information about safety *)
    let acc_check_div (t: t) (loc: Loc.t) (b: bool): t =
      if t.si_no_div0 != None && dbg_loc then
        F.printf "Loc:Set:Div at %a: %b\n" Loc.fp loc b;
      if b then t
      else
        match t.si_no_div0 with
        | None -> t
        | Some s ->
            { t with si_no_div0 = Some (SLoc.add loc s) }
    let acc_par_ok (t: t) (loc: Loc.t) (b: bool): t =
      if t.si_pars_ok != None && dbg_loc then
        F.printf "Loc:Set:Par at %a: %b\n" Loc.fp loc b;
      if b then t
      else
        match t.si_pars_ok with
        | None -> t
        | Some s ->
            { t with si_pars_ok = Some (SLoc.add loc s) }
  end
