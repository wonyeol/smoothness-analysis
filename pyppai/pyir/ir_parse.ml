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
 ** ir_parse.ml: entry point for frontend *)
open Analysis_sig
open Lib

(** General debug *)
let output = ref true

(** General parsing/processing function *)
let parse_code
    ~(use_pyast: bool) (* whether to use the external pyast *)
    ~(inline_fns: string list) (* funcs to be inlined *)
    (i: int option) (input: analysis_input): Ir_sig.prog =
  (* get python_code *)
  let python_code : string =
      match input with
      | AI_pyfile fname -> read_file fname
      | AI_pystring str -> str in
  (* construction of the Py.Object.t using Pyml *)
  let pyobj = Pyobj_util.get_ast python_code in
  let irast =
    if use_pyast then
      let pyast = Pyast_transfo.pyobj_to_modl pyobj in
      (* inline function definitions *)
      let pyast = Pyastl_util.inline_funcdef inline_fns pyast in
      (* construction of the IR AST *)
      Ir_cast.modl_to_prog pyast
    else
      (* construction of the Pyast local AST *)
      let pyast = Pyastl_cast.pyobj_to_modl pyobj in
      (* inline function definitions *)
      let pyast = Pyastl_util.inline_funcdef inline_fns pyast in
      (* construction of the IR AST *)
      Ir_cast.modl_to_prog pyast in
  if !output then
    begin
      let s =
        match i with
        | None -> ""
        | Some i -> Printf.sprintf "(%d)" i in
      Printf.printf "[IR%s]\n%a\n" s Ir_util.pp_prog irast
    end;
  irast
