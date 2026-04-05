from __future__ import annotations

import queue
import io
import sys
import threading
import tkinter as tk
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from . import cli as cli_module
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bili_music_list.cli as cli_module


class MusicListGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Bili Music List GUI")
        self.geometry("980x700")
        self.resizable(True, True)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        export_tab = ttk.Frame(notebook, padding=12)
        ai_tab = ttk.Frame(notebook, padding=12)
        notebook.add(export_tab, text="导出收藏夹 / Export")
        notebook.add(ai_tab, text="CSV AI 清洗 / AI Refine")

        self._build_export_tab(export_tab)
        self._build_ai_tab(ai_tab)

        log_frame = ttk.LabelFrame(self, text="运行日志 / Runtime Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.log_text = tk.Text(log_frame, height=14, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_export_tab(self, parent: ttk.Frame) -> None:
        self.export_media_id = tk.StringVar()
        self.export_parser = tk.StringVar(value="heuristic")
        self.export_output = tk.StringVar(value="output/music_list.csv")
        self.export_cookie_text = tk.StringVar()
        self.export_cookie_file = tk.StringVar()
        self.export_with_detail = tk.BooleanVar(value=False)
        self.export_include_unmatched = tk.BooleanVar(value=False)
        self.export_timeout = tk.StringVar(value="30")

        self.export_llm_base = tk.StringVar()
        self.export_llm_key = tk.StringVar()
        self.export_llm_model = tk.StringVar()
        self.export_llm_batch_size = tk.StringVar(value="10")
        self.export_llm_retries = tk.StringVar(value="3")
        self.export_llm_delay_ms = tk.StringVar(value="800")
        self.export_llm_max_tokens = tk.StringVar(value="800")

        row = 0
        self._add_entry(parent, "media_id", self.export_media_id, row)
        row += 1
        self._add_entry(parent, "output", self.export_output, row, file_save=True)
        row += 1
        self._add_entry(parent, "cookie (text)", self.export_cookie_text, row, secret=True)
        row += 1
        self._add_entry(parent, "cookie_file", self.export_cookie_file, row, file_open=True)
        row += 1

        ttk.Label(parent, text="parser").grid(row=row, column=0, sticky="w", pady=4)
        parser_box = ttk.Combobox(parent, textvariable=self.export_parser, values=["heuristic", "llm", "hybrid"], state="readonly")
        parser_box.grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Checkbutton(parent, text="with_detail", variable=self.export_with_detail).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Checkbutton(parent, text="include_unmatched", variable=self.export_include_unmatched).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        self._add_entry(parent, "timeout", self.export_timeout, row)
        row += 1

        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        ttk.Label(parent, text="LLM (可选 / optional)", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        self._add_entry(parent, "llm_base_url", self.export_llm_base, row)
        row += 1
        self._add_entry(parent, "llm_api_key", self.export_llm_key, row, secret=True)
        row += 1
        self._add_entry(parent, "llm_model", self.export_llm_model, row)
        row += 1
        self._add_entry(parent, "llm_batch_size", self.export_llm_batch_size, row)
        row += 1
        self._add_entry(parent, "llm_retries", self.export_llm_retries, row)
        row += 1
        self._add_entry(parent, "llm_delay_ms", self.export_llm_delay_ms, row)
        row += 1
        self._add_entry(parent, "llm_max_tokens", self.export_llm_max_tokens, row)
        row += 1

        btn = ttk.Button(parent, text="开始导出 / Run Export", command=self._run_export)
        btn.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        row += 1

        export_example = (
            "示例说明 / Example\n"
            "- 解析为csv文件，设置media-id即可，需要先将收藏夹设置为公开，或者附带上cookie参数\n"
            "  --media-id 123456 --parser heuristic --output output/music_list.csv"
        )
        ttk.Label(
            parent,
            text=export_example,
            justify=tk.LEFT,
            foreground="#444444",
            wraplength=860,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))

        parent.columnconfigure(1, weight=1)

    def _build_ai_tab(self, parent: ttk.Frame) -> None:
        self.ai_input_csv = tk.StringVar(value="output/music_list.csv")
        self.ai_output_csv = tk.StringVar(value="output/music_list_ai.csv")
        self.ai_llm_base = tk.StringVar(value="https://api.siliconflow.cn/v1")
        self.ai_llm_key = tk.StringVar()
        self.ai_llm_model = tk.StringVar(value="Qwen/Qwen2.5-7B-Instruct")
        self.ai_batch_size = tk.StringVar(value="8")
        self.ai_retries = tk.StringVar(value="1")
        self.ai_delay_ms = tk.StringVar(value="200")
        self.ai_max_tokens = tk.StringVar(value="250")
        self.ai_timeout = tk.StringVar(value="60")

        row = 0
        self._add_entry(parent, "input_csv", self.ai_input_csv, row, file_open=True)
        row += 1
        self._add_entry(parent, "ai_output", self.ai_output_csv, row, file_save=True)
        row += 1
        self._add_entry(parent, "llm_base_url", self.ai_llm_base, row)
        row += 1
        self._add_entry(parent, "llm_api_key", self.ai_llm_key, row, secret=True)
        row += 1
        self._add_entry(parent, "llm_model", self.ai_llm_model, row)
        row += 1
        self._add_entry(parent, "llm_batch_size", self.ai_batch_size, row)
        row += 1
        self._add_entry(parent, "llm_retries", self.ai_retries, row)
        row += 1
        self._add_entry(parent, "llm_delay_ms", self.ai_delay_ms, row)
        row += 1
        self._add_entry(parent, "llm_max_tokens", self.ai_max_tokens, row)
        row += 1
        self._add_entry(parent, "timeout", self.ai_timeout, row)
        row += 1

        btn = ttk.Button(parent, text="开始AI清洗 / Run AI Refine", command=self._run_ai_refine)
        btn.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        row += 1

        ai_example = (
            "示例说明 / Example\n"
            "- 如果csv中不准，还可以使用ai进一步解析csv文件，示例用的国内模型快速解析，只需要key换成自己的就行\n"
            "  --input-csv output/music_list.csv --ai-output output/music_list_ai.csv --llm-base-url "
            "https://api.siliconflow.cn/v1 --llm-model Qwen/Qwen2.5-7B-Instruct --llm-batch-size "
            "8 --llm-retries 1 --llm-delay-ms 200 --llm-max-tokens 250 --timeout 45 --llm-api-key YOUR_KEY"
        )
        ttk.Label(
            parent,
            text=ai_example,
            justify=tk.LEFT,
            foreground="#444444",
            wraplength=860,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))

        parent.columnconfigure(1, weight=1)

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        row: int,
        file_open: bool = False,
        file_save: bool = False,
        secret: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var, show="*" if secret else "")
        entry.grid(row=row, column=1, sticky="ew", pady=4, padx=(8, 8))
        if file_open:
            ttk.Button(parent, text="选择", command=lambda: self._pick_file(var)).grid(row=row, column=2, pady=4)
        elif file_save:
            ttk.Button(parent, text="保存到", command=lambda: self._pick_save(var)).grid(row=row, column=2, pady=4)

    def _pick_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _pick_save(self, var: tk.StringVar) -> None:
        path = filedialog.asksaveasfilename()
        if path:
            var.set(path)

    def _run_export(self) -> None:
        if not self.export_media_id.get().strip():
            messagebox.showerror("参数错误", "media_id 不能为空")
            return
        args = [
            "--media-id",
            self.export_media_id.get().strip(),
            "--parser",
            self.export_parser.get().strip(),
            "--output",
            self.export_output.get().strip(),
            "--timeout",
            self.export_timeout.get().strip(),
            "--llm-batch-size",
            self.export_llm_batch_size.get().strip(),
            "--llm-retries",
            self.export_llm_retries.get().strip(),
            "--llm-delay-ms",
            self.export_llm_delay_ms.get().strip(),
            "--llm-max-tokens",
            self.export_llm_max_tokens.get().strip(),
        ]
        if self.export_cookie_text.get().strip():
            args.extend(["--cookie", self.export_cookie_text.get().strip()])
        if self.export_cookie_file.get().strip():
            args.extend(["--cookie-file", self.export_cookie_file.get().strip()])
        if self.export_with_detail.get():
            args.append("--with-detail")
        if self.export_include_unmatched.get():
            args.append("--include-unmatched")
        self._append_llm_common(args, self.export_llm_base, self.export_llm_key, self.export_llm_model)
        self._start_cli_process(args)

    def _run_ai_refine(self) -> None:
        if not self.ai_input_csv.get().strip():
            messagebox.showerror("参数错误", "input_csv 不能为空")
            return
        args = [
            "--input-csv",
            self.ai_input_csv.get().strip(),
            "--ai-output",
            self.ai_output_csv.get().strip(),
            "--llm-base-url",
            self.ai_llm_base.get().strip(),
            "--llm-api-key",
            self.ai_llm_key.get().strip(),
            "--llm-model",
            self.ai_llm_model.get().strip(),
            "--llm-batch-size",
            self.ai_batch_size.get().strip(),
            "--llm-retries",
            self.ai_retries.get().strip(),
            "--llm-delay-ms",
            self.ai_delay_ms.get().strip(),
            "--llm-max-tokens",
            self.ai_max_tokens.get().strip(),
            "--timeout",
            self.ai_timeout.get().strip(),
        ]
        if not self.ai_llm_base.get().strip() or not self.ai_llm_key.get().strip() or not self.ai_llm_model.get().strip():
            messagebox.showerror("参数错误", "AI 模式需要填写 llm_base_url / llm_api_key / llm_model")
            return
        self._start_cli_process(args)

    def _append_llm_common(
        self,
        args: list[str],
        base: tk.StringVar,
        key: tk.StringVar,
        model: tk.StringVar,
    ) -> None:
        if base.get().strip() and key.get().strip() and model.get().strip():
            args.extend(
                [
                    "--llm-base-url",
                    base.get().strip(),
                    "--llm-api-key",
                    key.get().strip(),
                    "--llm-model",
                    model.get().strip(),
                ]
            )

    def _start_cli_process(self, cli_args: list[str]) -> None:
        if self.is_running:
            messagebox.showwarning("运行中", "已有任务在运行，请等待当前任务完成。")
            return

        self._log(f"$ bili-music-list {' '.join(cli_args)}")
        self.is_running = True
        self._set_buttons_state("disabled")

        def worker() -> None:
            try:
                out = _QueueWriter(self.log_queue)
                with redirect_stdout(out), redirect_stderr(out):
                    try:
                        cli_module.run_with_args(cli_args)
                        self.log_queue.put("[exit] code=0")
                    except SystemExit as exc:
                        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
                        self.log_queue.put(f"[exit] code={code}")
                    except Exception:
                        traceback.print_exc()
                        self.log_queue.put("[exit] code=1")
            except Exception as exc:
                self.log_queue.put(f"[error] {exc}")
            finally:
                self.is_running = False
                self.log_queue.put("[ui] enable_buttons")

        threading.Thread(target=worker, daemon=True).start()

    def _log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line == "[ui] enable_buttons":
                self._set_buttons_state("normal")
                continue
            self._log(line)
        self.after(100, self._drain_log_queue)

    def _set_buttons_state(self, state: str) -> None:
        for widget in self.winfo_children():
            self._set_widget_state_recursive(widget, state)

    def _set_widget_state_recursive(self, widget: tk.Widget, state: str) -> None:
        if isinstance(widget, ttk.Button):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, state)


class _QueueWriter(io.TextIOBase):
    def __init__(self, q: queue.Queue[str]) -> None:
        self.q = q

    def write(self, s: str) -> int:
        if s:
            for line in s.rstrip("\n").splitlines():
                self.q.put(line)
        return len(s)

    def flush(self) -> None:
        return None


def main() -> None:
    app = MusicListGui()
    app.mainloop()


if __name__ == "__main__":
    main()
