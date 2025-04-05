import gradio as gr
import subprocess
import os
import toml
import glob
import time
import signal
import random
import os
import re
import argparse
from copy import deepcopy

from ui_utils import (
    textbox_from_config,
    checkbox_from_config,
    number_from_config,
    slider_from_config,
    dropdown_from_config,
)

import image_to_tags
import image_to_captions
import huggingface_hub
from PIL import Image
from pathlib import Path

root_dir = os.path.dirname(os.path.abspath(__file__))

# print(root_dir)

# LOG_FILE = "test.log"
# PID_FILE = "test.pid"

LOG_FILE = os.path.join(root_dir, "test.log")
PID_FILE = os.path.join(root_dir, "test.pid")


MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

defaults = {
    "model": "wd-eva02-large-tagger-v3",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


# === Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Gradio Training Config Tool")
    parser.add_argument(
        "--train_type",
        choices=["flux", "sdxl"],
        default="flux",
        help="Loại mô hình training (flux hoặc sdxl)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Tuỳ chọn file TOML thay vì config_{train_type}.toml",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="only for sdxl",
    )

    parser.add_argument(
        "--share",
        default=False,
        type=bool,
        help="public gradio",
    )
    return parser.parse_args()


# === Config loader ===
def load_config(train_type="flux", config_path=None):
    try:
        if config_path and os.path.exists(config_path):
            print(f"🔧 Đang load config: {config_path}")
            return toml.load(config_path)

        user_path = os.path.join(root_dir, f"config_user_{train_type}.toml")

        base_path = os.path.join(root_dir, f"config_{train_type}.toml")

        if os.path.exists(user_path):
            print(f"🔧 Đang load: {user_path}")
            return toml.load(user_path)
        elif os.path.exists(base_path):
            print(f"🔧 Đang load: {base_path}")
            return toml.load(base_path)
        else:
            print("❌ Không tìm thấy file config")
            return {}
    except Exception as e:
        print(f"❌ Lỗi khi load config: {e}")
        return {}


def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)


# Hàm đọc cấu hình từ file TOML
def read_toml_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return toml.load(file)
    else:
        print(f"File {file_path} không tồn tại!")
        return None


# Hàm để đọc subnet từ dataset_config.toml và lấy prompt từ image_dir
def update_prompt_with_subsets(dataset_config, num_prompts):
    # Đọc cấu hình dataset_config
    # dataset_config = read_toml_file(dataset_config_path)
    # Đoạn mã tạo cấu hình prompt mặc định

    if train_type == "flux":

        prompt_config = {
            "prompt": {
                "width": 1024,
                "height": 1024,
                "scale": 3.5,
                "sample_steps": 28,
                "subset": [],  # Sẽ được cập nhật với các subsets từ dataset_config
            }
        }
    else:
        prompt_config = {
            "prompt": {
                "width": 758,
                "height": 1024,
                "scale": 7,
                "sample_steps": 28,
                "subset": [],  # Sẽ được cập nhật với các subsets từ dataset_config
            }
        }

    if dataset_config and "datasets" in dataset_config:
        subsets = []

        # Duyệt qua datasets và lấy subsets
        for dataset in dataset_config["datasets"]:
            if "subsets" in dataset:
                for subset in dataset["subsets"]:
                    image_dir = subset.get("image_dir")
                    caption_extension = subset.get("caption_extension")
                    # Giới hạn số lượng prompt theo giá trị num_prompts từ giao diện người dùng
                    if image_dir and caption_extension:
                        # Tìm các tệp caption trong image_dir có phần mở rộng là caption_extension
                        search_pattern = os.path.join(
                            image_dir, "**/*" + caption_extension
                        )
                        caption_files = glob.glob(search_pattern, recursive=True)

                        # Nếu có số lượng prompt giới hạn, chọn ngẫu nhiên
                        selected_files = random.sample(
                            caption_files, min(num_prompts, len(caption_files))
                        )

                        # Lấy các prompt từ các tệp caption
                        prompts = []
                        for file in selected_files:
                            with open(file, "r") as f:
                                prompts.append(f.read().strip())

                        # Thêm prompt vào config
                        for prompt in prompts:
                            new_prompt = {"prompt": prompt}
                            prompt_config["prompt"]["subset"].append(new_prompt)

                        # Cập nhật subset với số lượng prompt lấy được
                        # subset["num_prompts_taken"] = len(prompts)
                        subsets.append(subset)

    return prompt_config


def parse_folder_name(folder_name):
    try:
        folder_name_parts = folder_name.split("_")
        # print(folder_name_parts)
        if len(folder_name_parts) > 1:
            if folder_name_parts[0].isdigit():
                num_repeats = int(folder_name_parts[0])
                class_token = "_".join(folder_name_parts[1:])
                return num_repeats, class_token
        return None, None
    except Exception as e:
        print(f"\n❌ Error reading log: {e}")


def find_image_files(path):
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return [
        file
        for file in glob.glob(path + "/**/*", recursive=True)
        if file.lower().endswith(supported_extensions)
    ]


def process_data_dir2(data_dir, num_repeats, caption_extension, is_reg=False):
    subsets = []

    if not os.path.isdir(data_dir):
        return subsets

    all_subfolders = get_subfolders_or_self(data_dir)

    for folder in all_subfolders:

        folder_name = os.path.basename(folder)
        repeats, class_token = parse_folder_name(folder_name)

        # print(repeats, class_token)

        if repeats:
            num_repeats = repeats

        # print(num_repeats)

        images = find_image_files(folder)
        if not images:
            continue  # Bỏ qua nếu không có ảnh

        # Lấy danh sách các file caption của thư mục
        txt_caption_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        caption_caption_files = [
            f for f in os.listdir(folder) if f.endswith(".caption")
        ]

        # Kiểm tra và tạo các subnet
        if caption_extension == "all":
            # Nếu thư mục có cả .txt và .caption, tạo 2 subnet
            if txt_caption_files:
                subsets.append(
                    create_subset(
                        folder,
                        num_repeats,
                        class_token,
                        ".txt",
                        is_reg,
                    )
                )
            if caption_caption_files:
                subsets.append(
                    create_subset(
                        folder,
                        num_repeats,
                        class_token,
                        ".caption",
                        is_reg,
                    )
                )
        elif caption_extension == ".txt" and txt_caption_files:
            # Chỉ tạo subnet cho .txt
            subsets.append(
                create_subset(
                    folder,
                    num_repeats,
                    class_token,
                    ".txt",
                    is_reg,
                )
            )
        elif caption_extension == ".caption" and caption_caption_files:
            # Chỉ tạo subnet cho .caption
            subsets.append(
                create_subset(
                    folder,
                    num_repeats,
                    class_token,
                    ".caption",
                    is_reg,
                )
            )

    return subsets


def process_data_dir(data_dir, num_repeats, caption_extension, is_reg=False):
    if not os.path.isdir(data_dir):
        return []

    subsets = []
    all_subfolders = get_subfolders_or_self(data_dir)
    # print(all_subfolders)

    for folder in all_subfolders:
        folder_name = os.path.basename(folder)
        repeats, class_token = parse_folder_name(folder_name)
        current_repeats = repeats or num_repeats

        if not find_image_files(folder):
            continue

        # Kiểm tra caption có tồn tại không
        caption_types = {
            ".txt": any(f.endswith(".txt") for f in os.listdir(folder)),
            ".caption": any(f.endswith(".caption") for f in os.listdir(folder)),
        }

        # Nếu không có file caption nào thì bỏ qua thư mục
        if not any(caption_types.values()):
            continue

        # Xác định các định dạng cần xử lý
        extensions_to_process = (
            [".txt", ".caption"]
            if caption_extension == "all"
            else (
                [caption_extension]
                if caption_types.get(caption_extension, False)
                else []
            )
        )

        for ext in extensions_to_process:
            if caption_types[ext]:
                subsets.append(
                    create_subset(folder, current_repeats, class_token, ext, is_reg)
                )

    return subsets


def create_subset(folder_path, num_repeats, class_token, caption_extension, is_reg):

    subset = {
        "image_dir": folder_path,
        "num_repeats": num_repeats,
        "caption_extension": caption_extension,
    }

    if class_token != None:
        subset["class_tokens"] = class_token

    if is_reg:
        subset["is_reg"] = True
    return subset


def get_subfolders_or_self(folder_path):
    all_folders = []

    for dirpath, dirnames, _ in os.walk(folder_path):
        for dirname in dirnames:
            all_folders.append(os.path.join(dirpath, dirname))

    return all_folders if all_folders else [folder_path]


def on_generate_dataset_config(
    train_data_dir,
    reg_data_dir,
    caption_extension,
    resolution,
    flip_aug,
    keep_tokens,
    num_repeats,
    num_prompts,
):

    config_dir = os.path.join(root_dir, "config")
    # print(config_dir)
    # Cập nhật gọi hàm với caption_extension
    train_subsets = process_data_dir(train_data_dir, num_repeats, caption_extension)
    reg_subsets = process_data_dir(
        reg_data_dir, num_repeats, caption_extension, is_reg=True
    )

    subsets = train_subsets + reg_subsets

    config = {
        "general": {
            "enable_bucket": True,
            "keep_tokens": keep_tokens,
            "bucket_reso_steps": 64,
            "bucket_no_upscale": False,
        },
        "datasets": [
            {
                "resolution": resolution,
                "min_bucket_reso": 256,
                "max_bucket_reso": 1024,
                "flip_aug": flip_aug,
                "color_aug": False,
                "face_crop_aug_range": None,
                "subsets": subsets,
            }
        ],
    }

    dataset_config = os.path.join(config_dir, "dataset_config.toml")
    # print(dataset_config)
    sample_config = os.path.join(config_dir, "sample_prompt.toml")

    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    config_str = toml.dumps(config)
    config_sample_str = toml.dumps(update_prompt_with_subsets(config, num_prompts))

    with open(dataset_config, "w", encoding="utf-8") as f:
        f.write(config_str)

    with open(sample_config, "w", encoding="utf-8") as f:
        f.write(config_sample_str)

    return config_str, config_sample_str


def is_flux_running():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "flux_train_network.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = result.stdout.strip()

        if not output:
            return False  # Không tìm thấy tiến trình

        pids = [int(pid) for pid in output.splitlines()]
        for pid in pids:
            if is_pid_alive(pid):
                return True

        return False  # Có PID nhưng không còn sống
    except Exception as e:
        print(f"❌ Lỗi kiểm tra flux_train_network: {e}")
        return False


def is_running(train_type):
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{train_type}_train_network.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = result.stdout.strip()

        if not output:
            return False  # Không tìm thấy tiến trình

        pids = [int(pid) for pid in output.splitlines()]
        for pid in pids:
            if is_pid_alive(pid):
                return True

        return False  # Có PID nhưng không còn sống
    except Exception as e:
        print(f"❌ Lỗi kiểm tra {train_type}_train_network: {e}")
        return False


def is_pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False  # PID không tồn tại
    except PermissionError:
        return True  # Tồn tại nhưng không có quyền
    else:
        return True  # Tồn tại và có quyền


def start_process(num_prompts):

    if num_prompts is None or num_prompts <= 0:
        btn = gr.update(interactive=False)
    else:
        btn = gr.update(interactive=True)
    # Kiểm tra PID đang chạy không

    if is_running(train_type):
        return (
            gr.update(interactive=False),
            btn,
            gr.update(interactive=False),
        )
    # Chạy test.py bằng nohup
    # cmd = f"nohup python3 test.py > {LOG_FILE} 2>&1 & echo $! > {PID_FILE}"

    cmd = f"nohup accelerate launch /content/kohya-trainer/{train_type}_train_network.py --dataset_config='/content/colab_ui/config/dataset_config.toml' --config_file='/content/colab_ui/config/config_final.toml' > {LOG_FILE} 2>&1 & echo $! > {PID_FILE}"

    subprocess.call(cmd, shell=True)

    global content, is_run
    is_run = True

    content = "🚀 Đã bắt đầu"

    return (
        gr.update(interactive=False),
        btn,
        gr.update(interactive=True),
    )


def stop_process():
    try:
        # os.kill(pid, signal.SIGTERM)
        result = subprocess.run(["pkill", "-f", "flux_train_network.py"])

        return (
            f"🛑 Đã dừng tiến trình",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    except Exception as e:
        return (
            "⚠️ Không có tiến trình đang chạy.",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


content = ""
is_run = False


def clear_spaces(s):
    return re.sub(r"\s+", " ", s.strip())


def stream_log():
    last_pos = 0
    global content
    # content += content
    while True:
        if not os.path.exists(LOG_FILE):
            yield content
            time.sleep(0.5)
            continue
        try:

            file_size = os.path.getsize(LOG_FILE)
            if last_pos > file_size:
                # File bị xoá hoặc rotate → reset
                last_pos = 0
                content = ""

            with open(LOG_FILE, "r") as f:
                f.seek(last_pos)
                lines = f.readlines()
                for line in lines:
                    content += clear_spaces(line) + "\n"
                # if lines:
                #     content += "".join(lines)

                last_pos = f.tell()

            yield content

        except Exception as e:
            yield content + f"\n❌ Error reading log: {e}"

        time.sleep(0.5)


stop_tags_flag = False
stop_captions_flag = False


def tags(dir_data, model_repo, tags_trigger, threshold, character_threshold):
    global stop_tags_flag
    stop_tags_flag = False
    content = ""

    def download_model(model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
        )
        return csv_path, model_path

    if not os.path.isdir(dir_data):
        gr.Warning(f"path '{dir_data} cannot be found.")
        raise FileNotFoundError(f"path '{dir_data} cannot be found.'")

    dir_files = os.listdir(dir_data)
    if len(dir_files) == 0:
        gr.Warning(f"No files in path '{dir_data}'.")
        raise FileNotFoundError(f"No files in path '{dir_data}'.")

    valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
    dir_files = [
        f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)
    ]

    # csv_path, model_path = download_model(model_repo)
    csv_path, model_path = download_model(model_repo)
    if dir_files:
        for image_path in dir_files:
            if stop_tags_flag:
                print("Process stopped.", flush=True)
                gr.Info("Process stopped ℹ️", duration=5)
                return content  # Dừng và trả về kết quả đã yield tớ

            additional_feature = image_to_tags.run_tags(
                dir_data,
                image_path,
                csv_path,
                model_path,
                general_threshold=threshold,
                character_threshold=character_threshold,
            )
            prompt = tags_trigger + ", " + additional_feature

            path_txt = image_path.split(".")[0]

            with open(os.path.join(dir_data, path_txt) + ".txt", "a") as the_file:
                the_file.write(prompt)
            # content += additional_feature + "\n"
            content += "🖼️" + image_path + ": 📜" + prompt + "\n"
            yield content
        content += " Hoàn thành"
        yield content


# Hàm để dừng quá trình
def captions(captions_models, dir_data, caps_trigger):
    from transformers import AutoProcessor, AutoModelForCausalLM

    global stop_captions_flag
    stop_captions_flag = False
    content = ""
    model_id = captions_models
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto"
        )
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if not os.path.isdir(dir_data):
        gr.Warning(f"path '{dir_data} cannot be found.'")
        raise FileNotFoundError(f"path '{dir_data} cannot be found.'")

    dir_files = os.listdir(dir_data)
    if len(dir_files) == 0:
        gr.Warning(f"No files in path '{dir_data}'.")
        raise FileNotFoundError(f"No files in path '{dir_data}'.")

    valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
    dir_files = [
        f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)
    ]
    if dir_files:
        for image_path in dir_files:
            if stop_tags_flag:
                print("Process stopped.", flush=True)
                gr.Info("Process stopped ℹ️", duration=5)
                return content  # Dừng và trả về kết quả đã yield tớ

            additional_feature = image_to_captions.run_captions(
                model, processor, dir_data, image_path
            )

            prompt = caps_trigger + ", " + additional_feature

            path_txt = image_path.split(".")[0]

            with open(os.path.join(dir_data, path_txt) + ".caption", "a") as the_file:
                the_file.write(prompt)
            # content += additional_feature + "\n"
            content += "🖼️" + image_path + ": 📜" + prompt + "\n"
            yield content
        content += " Hoàn thành"
        yield content


# Hàm để dừng quá trình
def stop_tags_process():
    global stop_tags_flag
    stop_tags_flag = True


def stop_captions_process():
    global stop_captions_flag
    stop_captions_flag = True


def load_uploaded_config(file):

    try:
        if file is None:
            return "❌ Chưa chọn file cấu hình.", *[gr.update() for _ in FIELD_MAP]

        with open(file.name, "r", encoding="utf-8") as f:
            config_data = toml.load(f)

        updates = []
        for path, component in FIELD_MAP.items():
            keys = path.split(".")
            val = config_data
            for k in keys:
                val = val.get(k, {})
            if isinstance(val, dict):  # Không phải giá trị cuối cùng
                updates.append(gr.update())
            else:
                updates.append(gr.update(value=val))
        # print(updates)

        gr.Info(f"Đã load cấu hình")
        return updates

    except Exception as e:
        print(f"❌ Lỗi khi load config: {e}")
        return {}
    # return f"✅ Đã load cấu hình từ {file.name}", *updates


def show_file_input():
    return gr.update(visible=True)


def deep_merge_dict(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def auto_reload_config(train_type):
    try:
        # Ưu tiên: final > user > base
        # final_path = os.path.join(root_dir, "config", "config_final.toml")
        user_path = os.path.join(root_dir, f"config_user_{train_type}.toml")

        base_path = os.path.join(root_dir, f"config_{train_type}.toml")

        if os.path.exists(user_path):
            path_to_load = user_path
        elif os.path.exists(base_path):
            path_to_load = base_path
        else:
            print("❌ Không tìm thấy bất kỳ file config nào.")
            return [gr.update() for _ in FIELD_INPUTS]

        print(f"📂 Auto load: {path_to_load}")
        with open(path_to_load, "r", encoding="utf-8") as f:
            config_data = toml.load(f)

        results = []
        for path in FIELD_PATHS:
            keys = path.split(".")
            current = config_data
            for k in keys:
                current = current.get(k, {})
            if isinstance(current, dict):
                results.append(gr.update())
            else:
                results.append(gr.update(value=current))

        return results

    except Exception as e:
        print(f"❌ Lỗi khi auto load config: {e}")
        return [gr.update() for _ in FIELD_INPUTS]


def auto_reload_stop():

    if is_running(train_type):
        return gr.update(interactive=True), gr.update(interactive=True)
    else:

        return gr.update(interactive=False), gr.update(interactive=False)


CONFIG = load_config()


# === Save config ===
def set_value_by_path(config: dict, path: str, value):
    keys = path.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def save_user_config(*values):
    try:

        base_config = toml.load(os.path.join(root_dir, f"config_{train_type}.toml"))

        config_user = deepcopy(base_config)

        for path, value in zip(FIELD_PATHS, values):
            set_value_by_path(config_user, path, value)

        config_file = os.path.join(root_dir, f"config_user_{train_type}.toml")

        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(config_user, f)
        gr.Info("Lưu thành côngℹ️", duration=5)
        return gr.update(value=config_file, visible=True)
    except Exception as e:
        gr.Warning("Lỗi khi lưu cấu hình", duration=5)
        return gr.update(value=None, visible=False)


def on_setup_button_click(text_encoder, num_prompts, resolution, *values):

    global content
    try:
        content = "🚀 Bắt đầu xử lý...\n"
        # print(content)
        config = toml.load(os.path.join(root_dir, f"config_{train_type}.toml"))
        # print(config)

        # print(os.path.join(root_dir,"config_{train_type}.toml"))
        config_final = deepcopy(config)
        content += "📦 Đã load file config.toml\n"
        # if(train_type=="flux"):
        # Gán lại các field từ UI
        for path, value in zip(FIELD_PATHS, values):
            set_value_by_path(config_final, path, value)

        # Điều kiện đặc biệt cho Flux + text_encoder
        if train_type.lower() == "flux":
            if config_final["optimizer_config"]["optimizer_type"] == "AdaFactor":
                config_final["optimizer_config"]["optimizer_args"] = [
                    "relative_step=False" "scale_parameter=False" "warmup_init=False"
                ]
                config_final["optimizer_config"][
                    "lr_scheduler"
                ] = "constant_with_warmup"

            if text_encoder:
                config_final["network_config"]["network_args"] = ["train_t5xxl=true"]
                config_final["training_config"]["network_train_unet_only"] = False
        else:
            config_final["training_config"][
                "pretrained_model_name_or_path"
            ] = model_path

        if config_final["log_config"]["wandb_api_key"] != None:
            config_final["log_config"]["log_with"] = "wandb"
            config_final["log_config"]["wandb_run_name"] = (
                train_type + "_lora_" + config_final["training_config"]["output_name"]
            )

        config_final["advanced_config"]["metadata_title"] = config_final[
            "training_config"
        ]["output_name"]

        content += "⚙️ Áp dụng cấu hình đặc biệt cho Flux + text_encoder\n"

        # Ghi file config_final
        config_dir = os.path.join(root_dir, "config")
        os.makedirs(config_dir, exist_ok=True)

        user_config = os.path.join(root_dir, f"config_user_{train_type}.toml")
        
        with open(user_config, "w", encoding="utf-8") as f:
            toml.dump(config_final, f)
        
                # Gọi on_generate_dataset_config
        dataset_str, sample_str = on_generate_dataset_config(
            config_final["dataset_config"]["train_data_dir"],
            config_final["dataset_config"]["reg_data_dir"],
            config_final["dataset_config"]["caption_extension"],
            resolution,
            config_final["dataset_config"]["flip_aug"],
            config_final["dataset_config"]["keep_tokens"],
            config_final["dataset_config"]["num_repeats"],
            num_prompts,
        )

        if (
            "sample_config" in config_final
            and "num_prompts" in config_final["sample_config"]
        ):
            del config_final["sample_config"]["num_prompts"]
        if (
            "training_config" in config_final
            and "text_encoder" in config_final["training_config"]
        ):
            del config_final["training_config"]["text_encoder"]
        
        if (
            "dataset_config" in config_final
        ):
            del config_final["dataset_config"]

        config_final_path = os.path.join(config_dir, "config_final.toml")
        with open(config_final_path, "w", encoding="utf-8") as f:
            toml.dump(config_final, f)
        content += f"💾 Đã lưu file cấu hình vào {config_final_path}\n"



        content += "✅ Hoàn tất setup. Cấu hình dataset đã được sinh.\n"

        yield (
            toml.dumps(config_final),
            dataset_str,
            sample_str,
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    except Exception as e:
        gr.Error("dd")
        yield "", "", "", gr.update(interactive=False), gr.update(interactive=False)


def load_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return f"❌ Không tìm thấy {path}"


def auto_load_configs():
    config_dir = os.path.join(root_dir, "config")

    try:
        if is_running(train_type):
            config_path = os.path.join(root_dir, f"config_user_{train_type}.toml")
            if not os.path.exists(config_path):
                config_path = os.path.join(root_dir, f"config_{train_type}.toml")

            dataset_path = os.path.join(config_dir, "dataset_config.toml")
            sample_path = os.path.join(config_dir, "sample_prompt.toml")

            config_str = load_file(config_path)
            dataset_str = load_file(dataset_path)
            sample_str = load_file(sample_path)

            return config_str, dataset_str, sample_str
        else:

            return gr.update(value=""), gr.update(value=""), gr.update(value="")

    except Exception as e:
        print(f"⚠️ Lỗi kiểm tra PID: {e}")
        return gr.update(value=""), gr.update(value=""), gr.update(value="")


def change_text_encoder(text_encoder):

    if text_encoder:
        return gr.Textbox(visible=True)
    else:
        return gr.Textbox(visible=False)


def check_demo(output_dir):
    from PIL import Image
    import os

    sample_folder = os.path.join(output_dir, "sample")
    config_dir = os.path.join(root_dir, "config")

    sample_config = os.path.join(config_dir, "sample_prompt.toml")

    data = toml.load(sample_config)
    subsets = data["prompt"]["subset"]

    num_prompts = len(subsets)

    image_files = sorted(
        [
            f
            for f in os.listdir(sample_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ],
        key=lambda x: os.path.getmtime(os.path.join(sample_folder, x)),
    )

    total_images = len(image_files)
    if total_images < num_prompts * 2:
        return (
            f"❌ Cần ít nhất {num_prompts * 2} ảnh trong sample/, hiện có {total_images}.",
            [],
            [],
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if total_images % num_prompts != 0:
        return (
            f"⚠️ Số ảnh ({total_images}) không chia hết cho num_prompts ({num_prompts}) → Có thể chưa render đủ batch.",
            [],
            [],
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # Tính batch mới nhất
    original_files = image_files[0:num_prompts]  # batch đầu
    sample_files = image_files[-num_prompts:]  # batch cuối

    original_images = [
        Image.open(os.path.join(sample_folder, f)) for f in original_files
    ]
    sample_images = [Image.open(os.path.join(sample_folder, f)) for f in sample_files]

    return (
        f"✅ So sánh batch đầu và batch mới nhất. Tổng ảnh: {total_images}",
        sample_images,
        original_images,
        gr.update(value=sample_images, visible=True),
        gr.update(visible=False),
    )


def combine_images(img1, img2):
    w = img1.width + img2.width
    h = max(img1.height, img2.height)
    combined = Image.new("RGB", (w, h))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    return combined


def on_gallery_click(evt: gr.SelectData, originals, samples):
    index = evt.index

    combined_image = combine_images(originals[index], samples[index])

    return gr.update(value=combined_image, visible=True)


def main(args):

    print(f"🚀 Running in train mode: {args.train_type}")

    with gr.Blocks() as demo:

        with gr.Tab("🧠 Training"):
            # gr.Markdown(f"🔧 Config đang dùng: `config_{train_type}.toml`")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Training Config")

                    output_name = textbox_from_config(
                        CONFIG,
                        "training_config.output_name",
                        "Out name",
                        info="Base name of the output model file.",
                    )
                    output_dir = textbox_from_config(
                        CONFIG,
                        "training_config.output_dir",
                        "Output Dir",
                        info="Directory where the model checkpoints will be saved.",
                    )
                    gradient_checkpointing = checkbox_from_config(
                        CONFIG,
                        "training_config.gradient_checkpointing",
                        "Gradient Checkpointing",
                        info="Enable gradient checkpointing to reduce memory usage.",
                    )
                    max_train_epochs = number_from_config(
                        CONFIG,
                        "training_config.max_train_epochs",
                        "Max Train Epochs",
                        info="Training Epochs",
                    )
                    save_every_n_epochs = number_from_config(
                        CONFIG,
                        "training_config.save_every_n_epochs",
                        "Save Every N Epochs",
                        info="save checkpoint every N epochs ",
                    )
                    train_batch_size = number_from_config(
                        CONFIG,
                        "training_config.train_batch_size",
                        "Train Batch Size",
                        info="Batch size used during training.",
                    )

                    gr.Markdown("## ⚙️ Advanced")

                    author = textbox_from_config(
                        CONFIG,
                        "advanced_config.author",
                        "Author",
                        # info="Save State for train next",
                    )

                    save_state = checkbox_from_config(
                        CONFIG,
                        "advanced_config.save_state",
                        "Save State",
                        info="Save State for train next",
                    )

                    resume = textbox_from_config(
                        CONFIG,
                        "advanced_config.resume",
                        "Resume",
                        info="Path saved previous State to train next",
                    )

                with gr.Column():
                    gr.Markdown("## ⚙️ Optimizer")
                    optimizer_type = dropdown_from_config(
                        CONFIG,
                        "optimizer_config.optimizer_type",
                        "Optimizer Type",
                        choices=[
                            "AdamW",
                            "AdamW8bit",
                            "Lion8bit",
                            "Lion",
                            "SGDNesterov",
                            "SGDNesterov8bit",
                            "DAdaptation(DAdaptAdamPreprint)",
                            "DAdaptAdaGrad",
                            "DAdaptAdam",
                            "DAdaptAdan",
                            "DAdaptAdanIP",
                            "DAdaptLion",
                            "DAdaptSGD",
                            "AdaFactor",
                        ],
                        info="Optimizer type: AdamW, AdamW8bit, PagedAdamW, etc.",
                    )
                    optimizer_args = textbox_from_config(
                        CONFIG,
                        "optimizer_config.optimizer_args",
                        "Optimizer Args",
                        info='Extra args like "weight_decay=0.01 betas=0.9,0.999"',
                    )
                    learning_rate = number_from_config(
                        CONFIG,
                        "optimizer_config.learning_rate",
                        "Learning Rate",
                        info="Base learning rate.",
                    )
                    # text_encoder = gr.Checkbox(label="Text Encoder", value=True)
                    text_encoder = checkbox_from_config(
                        CONFIG,
                        "training_config.text_encoder",
                        "Text Encoder",
                        info="Base learning rate.",
                    )

                    text_encoder_lr = number_from_config(
                        CONFIG,
                        "optimizer_config.text_encoder_lr",
                        "Text Encoder LR",
                        info="Learning rate for text encoder.",
                    )
                    lr_scheduler = dropdown_from_config(
                        CONFIG,
                        "optimizer_config.lr_scheduler",
                        "LR Scheduler",
                        choices=[
                            "constant_with_warmup",
                            "constant",
                            "linear",
                            "cosine",
                            "polynomial",
                            "cosine_with_restarts",
                        ],
                        info="Schedule type for learning rate.",
                    )
                    lr_warmup_steps = number_from_config(
                        CONFIG,
                        "optimizer_config.lr_warmup_steps",
                        "LR Warmup Steps",
                        info="Warmup steps (int or float ratio).",
                    )
                    gr.Markdown("## ⚙️ Logging")

                    wandb_api_key = textbox_from_config(
                        CONFIG,
                        "log_config.wandb_api_key",
                        "WandB API Key",
                        info="Weights & Biases API key for logging (leave blank to disable).",
                    )

                with gr.Column():
                    gr.Markdown("## Dataset Config")
                    train_data_dir = textbox_from_config(
                        CONFIG,
                        "dataset_config.train_data_dir",
                        "Train Data Directory",
                        info="Path to training images.",
                    )
                    reg_data_dir = textbox_from_config(
                        CONFIG,
                        "dataset_config.reg_data_dir",
                        "Regularization Data Directory",
                        info="(Optional) Path to class images for regularization.",
                    )
                    caption_extension = dropdown_from_config(
                        CONFIG,
                        "dataset_config.caption_extension",
                        "Caption Extension",
                        choices=[".txt", ".caption", "all"],
                        info="Extension of caption files.",
                    )

                    num_repeats = number_from_config(
                        CONFIG,
                        "dataset_config.num_repeats",
                        "Num repeats",
                        info="Có thể điều chỉnh repeat cho từng thư mục bằng cách đặt tên thư mục theo cú pháp {repeat}_{folder_name}",
                    )
                    resolution = slider_from_config(
                        CONFIG,
                        "sample_config.resolution",
                        "Resolution",
                        minimum=512,
                        maximum=1024,
                        step=128,
                        info="Training image resolution.",
                    )
                    flip_aug = checkbox_from_config(
                        CONFIG,
                        "dataset_config.flip_aug",
                        "Flip Augmentation",
                        info="Apply horizontal flip augmentation.",
                    )
                    keep_tokens = number_from_config(
                        CONFIG,
                        "dataset_config.keep_tokens",
                        "Keep Tokens",
                        info="Keep this number of tokens from the prompt start.",
                    )

                    gr.Markdown("## Sample Config")

                    sample_every_n_epochs = number_from_config(
                        CONFIG,
                        "sample_config.sample_every_n_epochs",
                        "Sample Every N Epochs",
                        info="Generate sample images every N epochs",
                    )
                    num_prompts = number_from_config(
                        CONFIG,
                        "sample_config.num_prompts",
                        "Number of Prompts to Sample",
                        info="How many prompts to sample.",
                    )

                with gr.Column():
                    gr.Markdown("## ⚙️ Network")
                    network_alpha = number_from_config(
                        CONFIG,
                        "network_config.network_alpha",
                        "Network Alpha",
                        info="Alpha value for scaling LoRA weights.",
                    )
                    network_dim = number_from_config(
                        CONFIG,
                        "network_config.network_dim",
                        "Network Dim",
                        info="Dimension of the LoRA network.",
                    )
                    network_args = textbox_from_config(
                        CONFIG,
                        "network_config.network_args",
                        "Network Args",
                        info="Extra network arguments (format: key=value).",
                    )

                    gr.Markdown("## Noise Control")
                    noise_offset = number_from_config(
                        CONFIG,
                        "noise_config.noise_offset",
                        "Noise Offset",
                        info="Amount of noise offset to apply (e.g., 0.1).",
                    )
                    adaptive_noise_scale = textbox_from_config(
                        CONFIG,
                        "noise_config.adaptive_noise_scale",
                        "Adaptive Noise Scale",
                        info="Noise scaled by mean absolute latent value.",
                    )
                    multires_noise_iterations = slider_from_config(
                        CONFIG,
                        "noise_config.multires_noise_iterations",
                        "Multires Noise Iterations",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        info="How many iterations to apply multires noise.",
                    )
                    multires_noise_discount = slider_from_config(
                        CONFIG,
                        "noise_config.multires_noise_discount",
                        "Multires Noise Discount",
                        minimum=1,
                        maximum=10,
                        step=1,
                        info="Discount multiplier for multires noise.",
                    )
            with gr.Row():
                setup_btn = gr.Button("🚀Setup config")
                with gr.Group():
                    save_btn = gr.Button("💾 Lưu cấu hình", interactive=False)
                    download_btn = gr.File(label="📄 Tải file cấu hình", visible=False)

                with gr.Group():
                    load_btn = gr.Button("💾 Load cấu hình")
                    config_file_input = gr.File(
                        label="📤 Chọn file cấu hình TOML để load",
                        file_types=[".toml"],
                        visible=False,
                    )

            with gr.Row():

                config_preview = gr.Textbox(
                    label="Config Final (config_final.toml)", lines=20
                )
                dataset_preview = gr.Textbox(
                    label="Dataset Config (dataset_config.toml)", lines=20
                )
                sample_preview = gr.Textbox(
                    label="Sample Prompt Config (sample_prompt.toml)", lines=20
                )

            with gr.Row():
                start_btn = gr.Button("🚀 Bắt đầu", interactive=False)
                check_btn = gr.Button("Check demo", interactive=False)
                stop_btn = gr.Button("🛑 Stop", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "## 🧵 Gradio Streaming Log bằng `yield` (Không dùng thread)"
                    )
                    image_output = gr.Gallery(
                        label="Blur image", preview=False, visible=False
                    )
                    compare_output = gr.Image(
                        label="So sánh Gốc | Sample", visible=False
                    )

            with gr.Row():

                log_output = gr.Textbox(label="📄 Log (real-time)", lines=20)
        with gr.Tab("Captions and Tags"):
            # with gr.Row():
            gr.Markdown("## Tags")
            tags_data_dir = gr.Textbox(
                label="Data Directory", value="as/tag/5_aesthetic"
            )
            tags_trigger = gr.Textbox(value="", label=" Trigger word")

            tags_models = gr.Dropdown(
                choices=[
                    "SmilingWolf/wd-eva02-large-tagger-v3",
                    "SmilingWolf/wd-eva02-large-tagger-v3",
                    "SmilingWolf/wd-vit-large-tagger-v3",
                    "SmilingWolf/wd-swinv2-tagger-v3",
                    "SmilingWolf/wd-convnext-tagger-v3",
                    "SmilingWolf/wd-vit-tagger-v3",
                    "SmilingWolf/wd-v1-4-moat-tagger-v2",
                    "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
                    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
                    "SmilingWolf/wd-v1-4-vit-tagger-v2",
                ],
                label="Caption Extension",
                value="SmilingWolf/wd-eva02-large-tagger-v3",
            )
            threshold = gr.Slider(
                minimum=0, maximum=1, step=0.05, value=0.45, label="threshold"
            )
            character_threshold = gr.Slider(
                minimum=0, maximum=1, step=0.05, value=0.35, label="Resolution"
            )
            with gr.Row():
                start_tag_btn = gr.Button("Start tags")
                stop_tag_btn = gr.Button("Stop")
            tags_status = gr.Textbox(label="Tags Output", lines=10)
            start_tag_btn.click(
                fn=tags,
                inputs=[
                    tags_data_dir,
                    tags_models,
                    tags_trigger,
                    threshold,
                    character_threshold,
                ],
                outputs=tags_status,
            )
            stop_tag_btn.click(stop_tags_process, outputs=None)

            gr.Markdown("## Captions")
            captions_data_dir = gr.Textbox(
                label="Data Directory", value=root_dir + "/as"
            )
            caps_trigger = gr.Textbox(value="", label=" Trigger word")

            captions_models = gr.Dropdown(
                choices=[
                    "microsoft/Florence-2-large-ft",
                    "microsoft/Florence-2-large",
                    "microsoft/Florence-2-base-ft",
                    "microsoft/Florence-2-base",
                ],
                label="Caption Extension",
                value="microsoft/Florence-2-large-ft",
            )

            with gr.Row():
                start_cap_btn = gr.Button("🚀 Start caption")
                stop_cap_btn = gr.Button("🛑 Stop")
            captions_status = gr.Textbox(label="Captions Output", lines=10)
            start_cap_btn.click(
                fn=captions,
                inputs=[captions_models, captions_data_dir, caps_trigger],
                outputs=captions_status,
            )
            stop_cap_btn.click(stop_captions_process, outputs=None)

        # === Tab 1: Model ==

        # === Callback to build and show command ===

        global FIELD_MAP, FIELD_PATHS, FIELD_INPUTS

        FIELD_MAP = {
            "training_config.output_name": output_name,
            "training_config.output_dir": output_dir,
            "training_config.gradient_checkpointing": gradient_checkpointing,
            "training_config.max_train_epochs": max_train_epochs,
            "training_config.save_every_n_epochs": save_every_n_epochs,
            "training_config.train_batch_size": train_batch_size,
            "training_config.text_encoder": text_encoder,
            "log_config.wandb_api_key": wandb_api_key,
            "optimizer_config.optimizer_type": optimizer_type,
            "optimizer_config.optimizer_args": optimizer_args,
            "optimizer_config.learning_rate": learning_rate,
            "optimizer_config.text_encoder_lr": text_encoder_lr,
            "optimizer_config.lr_scheduler": lr_scheduler,
            "optimizer_config.lr_warmup_steps": lr_warmup_steps,
            "dataset_config.train_data_dir": train_data_dir,
            "dataset_config.reg_data_dir": reg_data_dir,
            "dataset_config.caption_extension": caption_extension,
            "dataset_config.num_repeats": num_repeats,
            "advanced_config.save_state": save_state,
            "advanced_config.resume": resume,
            "advanced_config.author": author,
            "dataset_config.flip_aug": flip_aug,
            "dataset_config.keep_tokens": keep_tokens,
            "sample_config.sample_every_n_epochs": sample_every_n_epochs,
            # "sample_config.sample_prompts": sample_prompts,
            # "flux_config.guidance_scale": guidance_scale,
            "sample_config.num_prompts": num_prompts,
            "network_config.network_alpha": network_alpha,
            "network_config.network_dim": network_dim,
            "network_config.network_args": network_args,
            "noise_config.noise_offset": noise_offset,
            "noise_config.adaptive_noise_scale": adaptive_noise_scale,
            "noise_config.multires_noise_iterations": multires_noise_iterations,
            "noise_config.multires_noise_discount": multires_noise_discount,
        }

        FIELD_INPUTS = list(FIELD_MAP.values())
        FIELD_PATHS = list(FIELD_MAP.keys())

        state_samples = gr.State()
        state_originals = gr.State()

        load_btn.click(fn=show_file_input, inputs=[], outputs=[config_file_input])

        setup_btn.click(
            # fn=on_setup_button_click,
            inputs=[text_encoder, num_prompts, resolution] + FIELD_INPUTS,
            fn=on_setup_button_click,
            outputs=[
                config_preview,
                dataset_preview,
                sample_preview,
                start_btn,
                save_btn,
            ],
        )

        text_encoder.change(
            fn=change_text_encoder,
            inputs=[text_encoder],
            outputs=text_encoder_lr,
        )
        config_file_input.change(
            fn=load_uploaded_config,
            inputs=[config_file_input],
            outputs=list(FIELD_MAP.values()),
        )
        check_btn.click(
            check_demo,
            inputs=[output_dir],
            outputs=[
                log_output,
                state_samples,
                state_originals,
                image_output,
                compare_output,
            ],
        )

        save_btn.click(
            fn=lambda *vals: save_user_config(*vals),
            inputs=FIELD_INPUTS,
            outputs=[download_btn],
        )

        image_output.select(
            fn=on_gallery_click,
            inputs=[state_originals, state_samples],
            outputs=[compare_output],
        )
        start_btn.click(
            start_process,
            inputs=[num_prompts],
            outputs=[start_btn, check_btn, stop_btn],
        )
        stop_btn.click(
            stop_process, outputs=[log_output, start_btn, check_btn, stop_btn]
        )

        demo.load(
            fn=lambda: auto_reload_config(train_type), inputs=[], outputs=FIELD_INPUTS
        )

        demo.load(fn=stream_log, outputs=[log_output])

        demo.load(
            fn=auto_load_configs,
            inputs=[],
            outputs=[config_preview, dataset_preview, sample_preview],
        )

        demo.load(
            fn=auto_reload_stop,
            inputs=[],
            outputs=[check_btn, stop_btn],
        )
        demo.launch(share=args.share)


if __name__ == "__main__":
    args = parse_args()
    global train_type, model_path
    train_type = args.train_type
    model_path = args.model_path
    main(args)
