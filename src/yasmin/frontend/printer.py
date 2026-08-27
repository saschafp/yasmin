from typing import Any

from yasmin.frontend import Operator
from yasmin.lowering.stencil_to_loop import lower


def print_loop_ir(node: Operator, indent: int = 3) -> None:
    # The Lower function can only handle Operators.
    if not isinstance(node, Operator):
        raise TypeError(f"Expected type Operator, got {type(node).__name__}")
        return

    indent_unit = " " * indent
    if hasattr(node, "_as_ir"):
        print(_rec_loop_ir_to_string(lower(node._as_ir()), 0, indent_unit))
    else:
        raise TypeError("Invalid input type. Function _as_ir is missing")


def _rec_loop_ir_to_string(node: Any, level: int = 0, indent_unit: str = " ") -> str:
    cls_name = type(node).__name__
    curr_ind = indent_unit * level
    next_ind = indent_unit * (level + 1)

    if cls_name == "Dimension":
        return f'Dimension(name="{node.name}")'
    elif cls_name == "Index":
        return f'Index(name="{node.name}")'
    elif cls_name == "Scalar":
        dtype_str = getattr(node.dtype, "name", str(node.dtype))
        return f'Scalar(name="{node.name}", dtype={dtype_str})'
    elif cls_name == "Literal":
        return f"Literal({node.value})"
    elif cls_name == "Field":
        dims_code = ", ".join(
            _rec_loop_ir_to_string(d, level + 1, indent_unit) for d in node.dims
        )
        comma = "," if len(node.dims) == 1 else ""
        dtype_str = getattr(node.dtype, "name", str(node.dtype))
        return (
            f"Field(\n"
            f"{next_ind}name='{node.name}',\n"
            f"{next_ind}dims=({dims_code}{comma}),\n"
            f"{next_ind}dtype={dtype_str},\n"
            f"{curr_ind})"
        )
    elif cls_name == "Extent":
        f_code = _rec_loop_ir_to_string(node.field, level + 1, indent_unit)
        return (
            f"Extent(\n"
            f"{next_ind}field={f_code},\n"
            f"{next_ind}dim={node.dim},\n"
            f"{curr_ind})"
        )
    elif cls_name == "BinaryExpr":
        op_name = node.op.name if hasattr(node.op, "name") else str(node.op)
        lhs_code = _rec_loop_ir_to_string(node.lhs, level + 1, indent_unit)
        rhs_code = _rec_loop_ir_to_string(node.rhs, level + 1, indent_unit)
        return (
            f"BinaryExpr(\n"
            f"{next_ind}BinaryOp.{op_name},\n"
            f"{next_ind}{lhs_code},\n"
            f"{next_ind}{rhs_code},\n"
            f"{curr_ind})"
        )
    elif cls_name == "Load":
        f_code = _rec_loop_ir_to_string(node.field, level + 1, indent_unit)
        idx_strs = [
            _rec_loop_ir_to_string(idx, level + 2, indent_unit) for idx in node.indices
        ]
        comma = "," if len(node.indices) == 1 else ""

        if idx_strs and not any("\n" in s for s in idx_strs):
            idx_tuple_str = f"({', '.join(idx_strs)}{comma})"
        else:
            idx_joined = (",\n" + indent_unit * (level + 2)).join(idx_strs)
            idx_tuple_str = (
                f"(\n{indent_unit * (level + 2)}{idx_joined}{comma}\n{next_ind})"
            )

        return (
            f"Load(\n"
            f"{next_ind}field={f_code},\n"
            f"{next_ind}indices={idx_tuple_str},\n"
            f"{curr_ind})"
        )
    elif cls_name == "Store":
        f_code = _rec_loop_ir_to_string(node.field, level + 1, indent_unit)
        idx_strs = [
            _rec_loop_ir_to_string(idx, level + 2, indent_unit) for idx in node.indices
        ]
        comma = "," if len(node.indices) == 1 else ""

        if idx_strs and not any("\n" in s for s in idx_strs):
            idx_tuple_str = f"({', '.join(idx_strs)}{comma})"
        else:
            idx_joined = (",\n" + indent_unit * (level + 2)).join(idx_strs)
            idx_tuple_str = (
                f"(\n{indent_unit * (level + 2)}{idx_joined}{comma}\n{next_ind})"
            )

        val_code = _rec_loop_ir_to_string(node.value, level + 1, indent_unit)
        return (
            f"Store(\n"
            f"{next_ind}field={f_code},\n"
            f"{next_ind}indices={idx_tuple_str},\n"
            f"{next_ind}value={val_code},\n"
            f"{curr_ind})"
        )
    elif cls_name == "For":
        idx_code = _rec_loop_ir_to_string(node.index, level + 1, indent_unit)
        lower_code = _rec_loop_ir_to_string(node.lower, level + 1, indent_unit)
        upper_code = _rec_loop_ir_to_string(node.upper, level + 1, indent_unit)

        body_stmts = getattr(node, "body", ())
        body_lines = [
            _rec_loop_ir_to_string(s, level + 2, indent_unit) for s in body_stmts
        ]
        comma = "," if len(body_stmts) == 1 else ""
        if body_lines:
            joined_body = (",\n" + indent_unit * (level + 2)).join(body_lines)
            body_str = (
                f"(\n{indent_unit * (level + 2)}{joined_body}{comma}\n{next_ind})"
            )
        else:
            body_str = "()"

        return (
            f"For(\n"
            f"{next_ind}index={idx_code},\n"
            f"{next_ind}lower={lower_code},\n"
            f"{next_ind}upper={upper_code},\n"
            f"{next_ind}body={body_str},\n"
            f"{curr_ind})"
        )
    elif cls_name == "Function":
        fields = getattr(node, "fields", ())
        scalars = getattr(node, "scalars", ())
        body_stmts = getattr(node, "body", ())

        field_lines = [
            _rec_loop_ir_to_string(f, level + 2, indent_unit) for f in fields
        ]
        comma_f = "," if len(fields) == 1 else ""
        joined_fields = (",\n" + indent_unit * (level + 2)).join(field_lines)
        fields_str = (
            f"(\n{indent_unit * (level + 2)}{joined_fields}{comma_f}\n{next_ind})"
            if fields
            else "()"
        )

        scalar_lines = [
            _rec_loop_ir_to_string(s, level + 2, indent_unit) for s in scalars
        ]
        comma_s = "," if len(scalars) == 1 else ""
        joined_scalars = (",\n" + indent_unit * (level + 2)).join(scalar_lines)
        scalars_str = (
            f"(\n{indent_unit * (level + 2)}{joined_scalars}{comma_s}\n{next_ind})"
            if scalars
            else "()"
        )

        body_lines = [
            _rec_loop_ir_to_string(s, level + 2, indent_unit) for s in body_stmts
        ]
        comma_b = "," if len(body_stmts) == 1 else ""
        joined_body = (",\n" + indent_unit * (level + 2)).join(body_lines)
        body_str = (
            f"(\n{indent_unit * (level + 2)}{joined_body}{comma_b}\n{next_ind})"
            if body_lines
            else "()"
        )

        return (
            f"Function(\n"
            f"{next_ind}name='{node.name}',\n"
            f"{next_ind}fields={fields_str},\n"
            f"{next_ind}scalars={scalars_str},\n"
            f"{next_ind}body={body_str},\n"
            f"{curr_ind})"
        )

    return str(node)


def print_stencil_ir(operator: Any, indent: int = 3):
    indent_unit = " " * indent
    if hasattr(operator, "_as_ir"):
        print(_rec_stercil_ir_to_string(operator._as_ir(), 0, indent_unit))
    else:
        raise TypeError("Invalid input type. Function _as_ir is missing.")
    return


def _rec_stercil_ir_to_string(node: Any, level: int = 0, indent_unit: str = " ") -> str:
    cls_name = type(node).__name__
    curr_ind = indent_unit * level
    next_ind = indent_unit * (level + 1)

    if cls_name in ("Dimension", "Field"):
        return node.name
    elif cls_name == "FieldAccess":
        field_ref = node.field.name if hasattr(node.field, "name") else str(node.field)
        return f"FieldAccess({field_ref}, {tuple(node.offsets)})"
    elif cls_name == "Literal":
        return f"Literal({node.value})"
    elif cls_name == "BinaryExpr":
        op_repr = (
            f"BinaryOp.{node.op.name}" if hasattr(node.op, "name") else str(node.op)
        )
        lhs_code = _rec_stercil_ir_to_string(node.lhs, level + 1, indent_unit)
        rhs_code = _rec_stercil_ir_to_string(node.rhs, level + 1, indent_unit)
        return (
            f"BinaryExpr(\n"
            f"{next_ind}{op_repr},\n"
            f"{next_ind}{lhs_code},\n"
            f"{next_ind}{rhs_code},\n"
            f"{curr_ind})"
        )
    elif cls_name == "Assign":
        tgt_code = _rec_stercil_ir_to_string(node.target, level + 1, indent_unit)
        val_code = _rec_stercil_ir_to_string(node.value, level + 1, indent_unit)
        return (
            f"Assign(\n"
            f"{next_ind}target={tgt_code},\n"
            f"{next_ind}value={val_code},\n"
            f"{curr_ind})"
        )
    elif cls_name == "Operator":
        stmts = getattr(node, "statements", [])
        stmt_lines = [
            _rec_stercil_ir_to_string(s, level + 2, indent_unit) for s in stmts
        ]
        joined_stmts = f",\n{indent_unit * (level + 2)}".join(stmt_lines)
        return (
            f"Operator(\n"
            f"{next_ind}statements=(\n"
            f"{indent_unit * (level + 2)}{joined_stmts},\n"
            f"{next_ind}),\n"
            f"{curr_ind})"
        )

    return str(node)
