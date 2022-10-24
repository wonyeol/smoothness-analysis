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


(** ***************)
(** string, print *)
(** ***************)
(* Constants into strings *)
val uop_to_string: uop -> string
val bop_to_string: bop -> string
val cop_to_string: cop -> string
val dist_kind_to_string:  dist_kind -> string
val dist_trans_to_string: dist_trans -> string

(* Functions to print into buffers *)
val buf_dist:   Buffer.t -> dist -> unit
val buf_number: Buffer.t -> number -> unit
val buf_expr:   Buffer.t -> expr -> unit
val buf_acmd:   Buffer.t -> acmd -> unit
val buf_stmt:   Buffer.t -> stmt -> unit
val buf_block:  Buffer.t -> block -> unit
val buf_prog:   Buffer.t -> prog -> unit

(* Conversion to strings *)
val number_to_string: number -> string
val expr_to_string:   expr -> string
val acmd_to_string:   acmd -> string
val prog_to_string:   prog -> string

(* Pretty-printing on channels *)
val pp_number:    out_channel -> number -> unit
val pp_expr:      out_channel -> expr -> unit
val pp_expr_list: out_channel -> expr list -> unit
val pp_acmd:      out_channel -> acmd -> unit
val pp_stmt:      out_channel -> stmt -> unit
val pp_block:     out_channel -> block -> unit
val pp_prog:      out_channel -> prog -> unit

(* Pretty-printing on formatters *)
val fp_dist:      form -> dist -> unit
val fp_number:    form -> number -> unit
val fp_expr:      form -> expr -> unit
val fp_expr_list: form -> expr list -> unit
val fp_acmd:      form -> acmd -> unit
val fp_stmt:      form -> stmt -> unit
val fp_block:     form -> block -> unit
val fp_prog:      form -> prog -> unit
val fpi_block:    string -> form -> block -> unit


(** ********************)
(** functions for expr *)
(** ********************)
(* to_{int,string}_opt *)
val expr_to_int_opt: expr -> int option
val expr_to_string_opt: expr -> string option

(* Function for modifying and simplifying expressions.
 * simplify_exp's argument performs an approximate type inference
 * of an expression *)
val simplify_exp: (expr -> exp_ty) -> expr -> expr


(** ********************)
(** functions for dist *)
(** ********************)
(* yes means true, but false means don't know *)
val dist_kind_support_subseteq: dist_kind -> dist_kind -> bool


(** ********************)
(** functions for stmt *)
(** ********************)
(* Checking various properties of statements *)
val contains_continue: stmt -> bool
val contains_break: stmt -> bool
(* val contains_return: stmt -> bool *)


(** ********************)
(** functions for prog *)
(** ********************)
(* Simplify use of Delta distributions in `sample` (not in `observe`) as follows:
 * replace each `Sample(trgt, _, (Delta, []), [arg], None)` by `Assn(trgt, arg)`. *)
val simplify_delta_prog: prog -> prog
  

(** ************************)
(** Apron helper functions *)
(** ************************)
(* Conversion of an IR expr into an Apron expression
 * (this function is very conservative and rejects many expressions) *)
val make_apron_expr: Apron.Environment.t -> expr -> Apron.Texpr1.t
val make_apron_cond: Apron.Environment.t -> expr -> Apron.Tcons1.t

(* Make thresholds to use for widening *)
val make_thr: Apron.Environment.t -> expr list -> Apron.Lincons1.earray


(** ***************************)
(** Analysis helper functions *)
(** ***************************)
(* Extract range info (int^3 opt) from args of range of plate. *)
val range_info_from_range: int option list -> (int * int * int) option
(* Extraction of range from plate *)
val range_info_from_plate: int option list -> (int * int * int) option
(* Range info for a loop *)
val range_info: exp_ty -> (int * int * int) option

(* Init, condition, increment for a for loop *)
val cond_of_for_loop: idtf -> (int * int * int) option -> stmt * expr * stmt
