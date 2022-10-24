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
 ** pyastl_util.ml: utilities on Python local AST *)
open Pyastl_sig


(** Checking various properties of statements *)
let rec contains_continue: 'a stmt -> bool = function
  | FunctionDef (_, _, body, _, _, _) ->
      (List.exists contains_continue body)
  | Return _  | Assign _ | AugAssign _->
      false
  | For(_, _, body, orelse, _) | While (_, body, orelse, _) | If (_, body, orelse, _) ->
      (List.exists contains_continue body) || (List.exists contains_continue orelse)
  | With (_, body, _) ->
      List.exists contains_continue body
  | Expr _ | Pass _  | Break _ ->
      false
  | Continue _ ->
      true

let rec contains_break: 'a stmt -> bool = function
  | FunctionDef (_, _, body, _, _, _) ->
      (List.exists contains_break body)
  | Return _  | Assign _ | AugAssign _ ->
      false
  | For(_, _, body, orelse, _) | While (_, body, orelse, _) | If (_, body, orelse, _) ->
      (List.exists contains_break body) || (List.exists contains_break orelse)
  | With (_, body, _) ->
      List.exists contains_break body
  | Expr _ | Pass _ ->
      false
  | Break _ ->
      true
  | Continue _ ->
      false

let rec contains_return: 'a stmt -> bool = function
  | FunctionDef (_, _, body, _, _, _) ->
      (List.exists contains_return body)
  | Return _  ->
      true
  | Assign _ | AugAssign _ ->
      false
  | For(_, _, body, orelse, _) | While (_, body, orelse, _) | If (_, body, orelse, _) ->
      (List.exists contains_return body) || (List.exists contains_return orelse)
  | With (_, body, _) ->
      List.exists contains_return body
  | Expr _ | Pass _  | Break _ | Continue _ ->
      false

let contains_middle_return (stmt: 'a stmt): bool =
  match stmt with
  | FunctionDef _ ->
     (* WL: I think having `return` in a funcdef is OK. *)
     (* contains_return stmt *)
     false
  | Return _ | Assign _ | AugAssign _ ->
     false
  | For _ | While _ | If _ | With _ ->
     contains_return stmt
  | Expr _  | Pass _ | Break _ | Continue _ ->
     false


(** Manipulate modules *)
let _inline_funcdef_stmt (name_l: string list):
      'a stmt -> 'a stmt list = function
  | FunctionDef (name, _, body, _, _, _)
       when (List.mem name name_l) -> body
  | stmt -> [stmt]

let _inline_funcdef_modl (name_l: string list):
      'a modl -> 'a modl = function
  | Module (stmt_l, a) ->
     let stmt_l_new =
       List.flatten (List.map (_inline_funcdef_stmt name_l) stmt_l) in
     Module (stmt_l_new, a)

let inline_funcdef = _inline_funcdef_modl


(*
(** Conversion to strings *)
let string_of_number = function
  | Int (n)      -> string_of_int n
  | Float (n)    -> string_of_float n
  (* | LongInt (n)  -> (string_of_int n) ^ "L" *)
  (* | Imag (n)     -> n *)
let string_of_boolop = function
  | And -> "and"
  | Or  -> "or"
let string_of_operator = function
  | Add         -> "+"
  | Sub         -> "-"
  | Mult        -> "*"
  | MatMult     -> "@"
  | Div         -> "/"
  | Mod         -> "%"
  | Pow         -> "**"
  | LShift      -> "<<"
  | RShift      -> ">>"
  | BitOr       -> "|"
  | BitXor      -> "^"
  | BitAnd      -> "&"
  | FloorDiv    -> "//"
let string_of_unaryop = function
  | Invert -> "~"
  | Not    -> "not"
  | UAdd   -> "+"
  | USub   -> "-"
let string_of_cmpop = function
  | Eq    -> "=="
  | NotEq -> "!="
  | Lt    -> "<"
  | LtE   -> "<="
  | Gt    -> ">"
  | GtE   -> ">="
  | Is    -> "is"
  | IsNot -> "is not"
  | In    -> "in"
  | NotIn -> "not in"
 *)
