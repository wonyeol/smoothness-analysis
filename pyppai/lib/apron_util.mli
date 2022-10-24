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
 ** apron_util.mli: parameterization with respect to Apron and utilities *)
open Apron
open Apron_sig

(** Available managers *)
module PA_box: APRON_MGR
module PA_oct: APRON_MGR
module PA_pol: APRON_MGR

(** Utilities to use Apron APIs *)
(* Creation of Apron variables *)
val make_apron_var: string -> Var.t
(* Coefficient extraction *)
val scalar_to_int: Mpfr.round -> Scalar.t -> int

(** Pretty-printing *)
val buf_linconsarray: string -> Buffer.t -> Lincons1.earray -> unit
