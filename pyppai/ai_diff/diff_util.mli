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

(** Some debugging flags *)
val dbg_apron:   bool
val dbg_init:    bool
val dbg_join:    bool
val dbg_compose: bool
val dbg_call:    bool
val dbg_param:   bool
val dbg_module:  bool
val dbg_sample:  bool
val dbg_distp:   bool
val dbg_observe: bool
val dbg_loop:    bool

(** Differentiability-related properties. *)
(* type for properties related to differentiability:
 *   Diff: differentiable w.r.t. some parameters S.
 *   Lips: locally Lipschitz w.r.t. some parameters S.
 *   Top:  always true. *)
type diff_prop = Diff | Lips | Top

(** Utilities for printing maps *)
val fp_diff_prop_short: form -> diff_prop -> unit
val fp_diff_prop: form -> diff_prop -> unit
val fp_for_diff_prop: form -> diff_prop -> unit
val ppm: (out_channel -> 'a -> unit) -> out_channel -> 'a SM.t -> unit
val fpm: (form -> 'a -> unit) -> form -> 'a SM.t -> unit
val fp_dist_par: form -> bool list option -> unit

(** Utilities for option type *)
val bind_opt: 'a option -> ('a -> 'b option) -> 'b option

(** Utilities for the operations in the diff domain *)
val map_join_union: ('a -> 'a -> 'a)
  -> 'a SM.t -> 'a SM.t -> 'a SM.t
val map_join_inter: ('a -> 'a -> 'a option)
  -> 'a SM.t -> 'a SM.t -> 'a SM.t
val map_equal: (string -> 'a -> bool) -> ('a -> 'a -> bool)
  -> 'a SM.t -> 'a SM.t -> bool
val lookup_with_default: string -> 'a SM.t -> 'a -> 'a

(** Preanalysis to get the set of parameters for which the differentiability
 ** analysis should provide results *)
val prog_varparams: getpars:bool -> getvars:bool -> prog -> SS.t

(** Abstraction of dependences in an expression *)
val dep_expr: expr -> SS.t

(** A very basic, hackish localisation setup *)
module Loc:
    sig
      type t
      val fp: form -> t -> unit
      val compare: t -> t -> int
      val start: t
      val next: t -> t
      val if_t: t -> t
      val if_f: t -> t
      val loop: t -> t
      val parn: t -> int -> t
    end

(** A basic domain to track safety information *)
module SI:
    sig
      type t
      (* General information *)
      val nowhere_div0: t -> bool
      val nowhere_par_ko: t -> bool
      (* Printing *)
      val pp: out_channel -> t -> unit
      val fp: form -> t -> unit
      (* No information *)
      val top: t
      (* Information for identity transformer *)
      val id: t
      (* Lattice operations *)
      val join: t -> t -> t
      val equal: t -> t -> bool
      (* Get information about safety *)
      val no_div0: t -> Loc.t -> bool
      val par_ok:  t -> Loc.t -> bool
      (* Accumulate information about safety *)
      val acc_check_div: t -> Loc.t -> bool -> t
      val acc_par_ok:    t -> Loc.t -> bool -> t
    end
