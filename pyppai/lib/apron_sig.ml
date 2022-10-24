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
 ** apron_sig.ml: parameterization with respect to Apron *)
open Apron

(** Signature for an Apron abstract domain manager wrapper
 **  (this is used to select Apron domain implementations) *)
module type APRON_MGR =
  sig
    val module_name: string
    type t
    val man: t Apron.Manager.t
  end
