import os


def tmpl_path(template):
    if not template.endswith(".j2"):
        template = f"{template}.j2"
    base, _ = os.path.split(__file__)
    tmpl_p = os.path.abspath(os.path.join(base, template))
    if os.path.exists(tmpl_p):
        return tmpl_p
    else:
        raise FileNotFoundError(f"Template {template} not found.")


def read_tmpl_str(template):
    with open(tmpl_path(template), "r") as f:
        lines = f.readlines()

    return "".join(lines)


if __name__ == "__main__":
    print(read_tmpl_str("tabular.tex"))
