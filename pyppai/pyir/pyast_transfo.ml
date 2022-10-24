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
 ** pyast_transfo.ml: conversion from Py.Object.t to external Pyast_sig *)

open Pyastl_sig

module PyAst = Pyast.Latest

let to_expr_context (context: PyAst.expr_context): expr_context =
  match context with
  | Load -> Load
  | Store -> Store
  | Del -> Del
(*
  | Param -> Param
*)

let to_boolop (boolop: PyAst.boolop): boolop =
  match boolop with
  | And -> And
  | Or -> Or

let to_operator (operator: PyAst.operator): operator =
  match operator with
  | Add -> Add
  | Sub -> Sub
  | Mult -> Mult
  | MatMult -> MatMult
  | Div -> Div
  | Mod -> Mod
  | Pow -> Pow
  | LShift -> LShift
  | RShift -> RShift
  | BitOr -> BitOr
  | BitXor -> BitXor
  | BitAnd -> BitAnd
  | FloorDiv -> FloorDiv

let to_unaryop (unaryop: PyAst.unaryop): unaryop =
  match unaryop with
  | Invert -> Invert
  | Not -> Not
  | UAdd -> UAdd
  | USub -> USub

let to_cmpop (cmpop: PyAst.cmpop): cmpop =
  match cmpop with
  | Eq -> Eq
  | NotEq -> NotEq
  | Lt -> Lt
  | LtE -> LtE
  | Gt -> Gt
  | GtE -> GtE
  | Is -> Is
  | IsNot -> IsNot
  | In -> In
  | NotIn -> NotIn

let rec to_modl (mod_: PyAst.mod_): 'a option modl =
  let a = None in
  match mod_ with
  | Module { body; _ } ->
     Module (List.map to_stmt body, a)
  | _ ->
     failwith (Format.asprintf "to_modl: not a module %a"
       (Refl.pp [%refl: PyAst.mod_] []) mod_)
and to_stmt (stmt: PyAst.stmt): 'a option stmt =
  let a = None in
  match stmt.desc with
  | FunctionDef { name; args; body; decorator_list; returns; _ } ->
     let args = to_arguments args in
     let body = List.map to_stmt body in
     let decorator_list = List.map to_expr decorator_list in
     let returns = Option.map to_expr returns in
     FunctionDef (name, args, body, decorator_list, returns, a)
  | Return value ->
     Return (Option.map to_expr value, a)
  | Assign { targets; value; _ } ->
     Assign (List.map to_expr targets, to_expr value, a)
  | AugAssign { target; op; value; } ->
     AugAssign (to_expr target, to_operator op, to_expr value, a)
  | For { target; iter; body; orelse; _ } ->
     let target = to_expr target in
     let iter = to_expr iter in
     let body = List.map to_stmt body in
     let orelse = List.map to_stmt orelse in
     For (target, iter, body, orelse, a)
  | While { test; body; orelse } ->
     let test = to_expr test in
     let body = List.map to_stmt body in
     let orelse = List.map to_stmt orelse in
     While (test, body, orelse, a)
  | If { test; body; orelse } ->
     let test = to_expr test in
     let body = List.map to_stmt body in
     let orelse = List.map to_stmt orelse in
     If (test, body, orelse, a)
  | With { items; body; _ } ->
     With (List.map to_withitem items, List.map to_stmt body, a)
  | Expr value ->
     Expr (to_expr value, a)
  | Pass -> Pass a
  | Break -> Break a
  | Continue -> Continue a
  | ClassDef _ ->
      (* hy: Ignore a class definition. This amounts to
       * making an assumption that only global-side-effect free functions
       * from the defined class are used. *)
      Pass a
  | Import _ | ImportFrom _ ->
     (* wl: Ignore `import ...` and `from ... import ...`. *)
     Pass a
  | _ ->
     failwith
       (Format.asprintf "to_stmt: unsupported statement %a"
         (Refl.pp [%refl: PyAst.stmt] []) stmt)

and to_expr (expr: PyAst.expr): 'a option expr =
  let a = None in
  match expr.desc with
  | BoolOp { op; values } ->
      BoolOp (to_boolop op, List.map to_expr values, a)
  | BinOp { left; op; right } ->
      BinOp (to_expr left, to_operator op, to_expr right, a)
  | UnaryOp { op; operand } ->
     UnaryOp (to_unaryop op, to_expr operand, a)
  | Dict { keys; values } ->
      Dict (List.map to_expr keys, List.map to_expr values, a)
  | Compare { left; ops; comparators } ->
      let left = to_expr left in
      let ops = List.map to_cmpop ops in
      let comparators = List.map to_expr comparators in
      Compare (left, ops, comparators, a)
  | Call { func; args; keywords } ->
      let func = to_expr func in
      let args = List.map to_expr args in
      let keywords = List.map to_keyword keywords in
      Call (func, args, keywords, a)
  | Constant { value; _ } ->
      begin match value with
      | None -> NameConstant (None, a)
      | Some (Bool b) -> NameConstant (Some b, a)
      | Some (Num (Int i)) -> Num (Int i, a)
      | Some (Num (Float f)) -> Num (Float f, a)
      | Some (Str s) -> Str (s, a)
      | Some (Ellipsis) -> Ellipsis (a)
      | _ ->
         failwith
           (Format.asprintf "to_expr: unsupported constant %a"
             (Refl.pp [%refl: PyAst.constant] []) value)
      end
  | Attribute { value; attr; ctx } ->
      Attribute (to_expr value, attr, to_expr_context ctx, a)
  | Subscript { value; slice; ctx } ->
      Subscript (to_expr value, to_slice slice, to_expr_context ctx, a)
  | Starred { value; ctx } ->
      Starred (to_expr value, to_expr_context ctx, a)
  | Name { id; ctx }  ->
      Name (id, to_expr_context ctx, a)
  | List { elts; ctx } ->
      List (List.map to_expr elts, to_expr_context ctx, a)
  | Tuple { elts; ctx } ->
      Tuple (List.map to_expr elts, to_expr_context ctx, a)
  | _ ->
      failwith (Format.asprintf "to_expr: unsupported expression %a"
        (Refl.pp [%refl: PyAst.expr] []) expr)

and to_slice (slice: PyAst.slice): 'a option slice =
  match slice.desc with
  | Slice { lower; upper; step } ->
      let lower = Option.map to_expr lower in
      let upper = Option.map to_expr upper in
      let step = Option.map to_expr step in
      Slice (lower, upper, step)
  | Tuple { elts = dims; _ } ->
      ExtSlice (List.map to_slice dims)
  | _ ->
      Index (to_expr slice)

and to_arguments (arguments: PyAst.arguments): 'a option arguments =
  let args = List.map to_arg arguments.args in
  let vararg = Option.map to_arg arguments.vararg in
  let kwonlyargs = List.map to_arg arguments.kwonlyargs in
  let kw_defaults = List.map to_expr arguments.kw_defaults in
  let kwarg = Option.map to_arg arguments.kwarg in
  let defaults = List.map to_expr arguments.defaults in
  (args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)

and to_arg (arg: PyAst.arg): 'a option arg =
  let a = None in
  (arg.arg, Option.map to_expr arg.annotation, a)

and to_keyword (keyword: PyAst.keyword): 'a option keyword =
  (keyword.arg, to_expr keyword.value)

and to_withitem (withitem: PyAst.withitem): 'a option withitem =
  (to_expr withitem.context_expr, Option.map to_expr withitem.optional_vars)

let pyobj_to_modl (obj: Py.Object.t): 'a option modl =
  to_modl (PyAst.parse_ast obj)
