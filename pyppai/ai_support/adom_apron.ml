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
 ** adom_apron.ml: Numerical abstract domain based on Apron *)
open Lib
open Ir_sig
open Ir_ty_sig
open Adom_sig
open Apron

open Apron_sig
open Apron_util

module IU = Ir_util


(** Apron domain constructor *)
module Make = functor (M: APRON_MGR) ->
  (struct
    let module_name = "Apron(" ^ M.module_name ^ ")"

    module A = Apron.Abstract1
    let man = M.man

    (* Abstract values:
     * - an enviroment
     * - and a conjunction of constraints in Apron representation (u) *)
    type t = M.t A.t

    let is_bot t = A.is_bottom man t

    (* Pp *)
    let buf_t (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "%a" (buf_linconsarray "") (A.to_lincons_array man t)
    let to_string = buf_to_string buf_t
    let pp = buf_to_channel buf_t

    (* tries to prove that a condition holds
     * soundness: if sat e t returns true, all states in gamma(t) satisfy e *)
    let sat (e: expr) (t: t): bool =
      let env = A.env t in
      let ce = Ir_util.make_apron_cond env e in
      A.sat_tcons man t ce

    (* Evaluation of commands *)
    let rec eval ac u =
      match ac with
      | Assert e ->
          if sat e u then
            u
          else
            failwith (Printf.sprintf "%s.eval: Cannot prove Assert" module_name)
      | Assume e ->
          (* convert the expression to Apron IR *)
          let env = A.env u in
          let ce = Ir_util.make_apron_cond env e in
          let eacons = Tcons1.array_make env 1 in
          Tcons1.array_set eacons 0 ce;
          (* perform the condition test *)
          let u = A.meet_tcons_array man u eacons in
          (* red bottom *)
          if is_bot u then raise Bottom
          else u

      | Assn (x,e) ->
          (* convert the expression to Apron IR *)
          let lv = make_apron_var x
          and rv = Ir_util.make_apron_expr (A.env u) e in
          (* perform the Apron assignment *)
          A.assign_texpr_array man u [| lv |] [| rv |] None

      | AssnCall _ ->
          failwith "todo:eval:assncall"

      | Sample _ ->
          failwith "todo:eval:sample"

    let enter_with withitem u =
      failwith "todo:enter_with:adom_apron"

    let exit_with withitem u =
      failwith "todo:exit_with:adom_apron"

    (* Lattice elements and operations *)
    let top =
      let env_empty = Environment.make [| |] [| |] in
      A.top man env_empty
    let init_t = top

    let join = A.join man
    let widen thr x0 x1 =
      if thr = [ ] then A.widening man x0 x1
      else A.widening_threshold man x0 x1 (Ir_util.make_thr (A.env x0) thr)
    let leq (u0: t) (u1: t): bool =
      A.is_leq man u0 u1

    (* Dimensions management *)
    (* these functions are not too hard to implement, but first we need
     * to make sure the interfaces are ok *)
    let dim_add (dn: string) (t: t): t =
      let var = make_apron_var dn in
      let env_old = A.env t in
      let env_new =
        try Environment.add env_old [| |] [| var |]
        with e ->
          failwith (Printf.sprintf "dim_add: %s" (Printexc.to_string e)) in
      A.change_environment man t env_new false
    let dim_rem_set (dns: SS.t) (t: t): t =
      let lvars =
        SS.fold (fun dn l -> make_apron_var dn :: l) dns [ ] in
      let env_old = A.env t in
      let env_new =
        try Environment.remove env_old (Array.of_list lvars)
        with e ->
          failwith (Printf.sprintf "dim_rem: %s" (Printexc.to_string e)) in
      A.change_environment man t env_new false
    let dim_rem (dn: string) (t: t): t =
      let var = make_apron_var dn in
      let env_old = A.env t in
      let env_new =
        try Environment.remove env_old [| var |]
        with e ->
          failwith (Printf.sprintf "dim_rem: %s" (Printexc.to_string e)) in
      A.change_environment man t env_new false
    let dim_mem (dn: string) (t: t): bool =
      let var = make_apron_var dn in
      let env = A.env t in
      try Environment.mem_var env var
      with e ->
        failwith (Printf.sprintf "dim_mem: %s" (Printexc.to_string e))
    let dim_project_out (dn: string) (t: t): t =
      let var = make_apron_var dn in
      A.forget_array man t [| var |] false
    let dims_get (t: t): SS.t option =
      let env = A.env t in
      let ai, af = Environment.vars env in
      let r = ref SS.empty in
      let f v = r := SS.add (Var.to_string v) !r in
      Array.iter f ai;
      Array.iter f af;
      Some !r

    (* ad-hoc function *)
    let set_aux_distty vto t =
      failwith "adom_apron.ml: set_aux_distty must not be called!"

    let bound_var_apron (dn: string) (t: t): int*int =
        let open Interval in
        let intvl: Interval.t = A.bound_variable man t (make_apron_var dn) in
        let inf : int = scalar_to_int Mpfr.Down intvl.inf in
        let sup : int = scalar_to_int Mpfr.Up   intvl.sup in
        (inf, sup)

    (* checks whether a domain-specific relationship holds
     * between two abstract states *)
    let is_related t1 t2 = false

    let range_info e t =
      failwith "todo:range_info:adom_apron"
  end: ABST_DOMAIN_NB_D)
