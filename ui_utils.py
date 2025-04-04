import gradio as gr
import toml

def get_value_by_path(config: dict, path: str, default=None):
    """
    Truy xuất giá trị từ dict lồng nhau với path như 'section.key.subkey'
    """
    try:
        keys = path.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    except Exception as e:
        print(f"[get_value_by_path] Error: {e}")
        return default


def textbox_from_config(config: dict, path: str, label: str, **kwargs):
    value = get_value_by_path(config, path, default="")
    # print("value")
    # print(config)
    return gr.Textbox(label=label, value=value, **kwargs)


def checkbox_from_config(config: dict, path: str, label: str, **kwargs):
    value = get_value_by_path(config, path, default=False)
    return gr.Checkbox(label=label, value=value, **kwargs)


def number_from_config(config: dict, path: str, label: str, **kwargs):
    value = get_value_by_path(config, path, default=0)
    return gr.Number(label=label, value=value, **kwargs)


def slider_from_config(config: dict, path: str, label: str, minimum=0, maximum=100, step=1, **kwargs):
    value = get_value_by_path(config, path, default=minimum)
    return gr.Slider(label=label, value=value, minimum=minimum, maximum=maximum, step=step, **kwargs)


def dropdown_from_config(config: dict, path: str, label: str, choices=None, **kwargs):
    if not choices:
        raise ValueError("Dropdown requires a non-empty 'choices' list.")
    value = get_value_by_path(config, path, default=choices[0])
    return gr.Dropdown(label=label, value=value, choices=choices, **kwargs)
