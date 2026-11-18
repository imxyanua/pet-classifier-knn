# code by xyanua - github.com/imxyanua
"""Desktop app: nhận dạng thú cưng bằng k-NN tự cài."""
import os
import threading
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox

import customtkinter as ctk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

from config import DISPLAY_NAMES, K_CHOICES, METRIC_LABELS, SELECTED_BREEDS
from dataset import load_feature_cache, load_or_extract
from features import features_and_preview, load_bgr, overlay_mask, segment_pet
from knn import (
    accuracy_score,
    confusion_matrix,
    knn_predict_batch,
    knn_predict_one,
    standardize_apply,
    standardize_fit,
    stratified_split,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def bgr_to_pil(bgr, max_side=380):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)


class PetClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Nhận dạng thú cưng · k-NN")
        self.geometry("1180x760")
        self.minsize(1020, 680)

        self.X = None
        self.y = None
        self.paths = None
        self.mean = None
        self.std = None
        self.train_idx = None
        self.test_idx = None
        self.current_path = None
        self.photo_refs = {}
        self.eval_running = False

        self._build()
        self.after(200, self._try_load_cache)

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        ctk.CTkLabel(
            header,
            text="Nhận dạng thú cưng bằng k-NN",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Tiền xử lý · phân đoạn · đặc trưng màu & hình dạng · k-NN tự cài",
            text_color=("gray40", "gray70"),
        ).pack(anchor="w")

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 8))
        self.tab_predict = self.tabs.add("Nhận dạng")
        self.tab_eval = self.tabs.add("Đánh giá")
        self._build_predict_tab()
        self._build_eval_tab()

        self.status = ctk.CTkLabel(self, text="Sẵn sàng.", anchor="w")
        self.status.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 12))

    def _build_predict_tab(self):
        tab = self.tab_predict
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(tab, width=320)
        side.grid(row=0, column=0, sticky="nsw", padx=(0, 12), pady=8)
        side.grid_propagate(False)

        ctk.CTkButton(side, text="Mở ảnh…", command=self.open_image).pack(
            fill="x", padx=14, pady=(16, 8)
        )
        ctk.CTkButton(side, text="Nhận dạng", command=self.predict_current).pack(
            fill="x", padx=14, pady=(0, 14)
        )

        ctk.CTkLabel(side, text="Số láng giềng k", anchor="w").pack(fill="x", padx=14)
        self.k_var = ctk.IntVar(value=5)
        self.k_label = ctk.CTkLabel(side, text="k = 5")
        self.k_label.pack(anchor="w", padx=14)
        self.k_slider = ctk.CTkSlider(
            side,
            from_=1,
            to=11,
            number_of_steps=10,
            command=self._on_k_change,
        )
        self.k_slider.set(5)
        self.k_slider.pack(fill="x", padx=14, pady=(0, 12))

        ctk.CTkLabel(side, text="Khoảng cách", anchor="w").pack(fill="x", padx=14)
        self.metric_seg = ctk.CTkSegmentedButton(
            side, values=list(METRIC_LABELS.keys())
        )
        self.metric_seg.set("Euclidean")
        self.metric_seg.pack(fill="x", padx=14, pady=(4, 16))

        self.result_title = ctk.CTkLabel(
            side, text="Chưa có dự đoán", font=ctk.CTkFont(size=18, weight="bold")
        )
        self.result_title.pack(anchor="w", padx=14, pady=(8, 4))
        self.neighbor_box = ctk.CTkTextbox(side, height=180)
        self.neighbor_box.pack(fill="both", expand=True, padx=14, pady=(0, 16))
        self.neighbor_box.insert("1.0", "Mở một ảnh rồi nhấn Nhận dạng.")
        self.neighbor_box.configure(state="disabled")

        images = ctk.CTkFrame(tab, fg_color="transparent")
        images.grid(row=0, column=1, sticky="nsew", pady=8)
        images.grid_columnconfigure((0, 1), weight=1)
        images.grid_rowconfigure(0, weight=1)

        self.orig_frame, self.orig_label = self._image_card(images, "Ảnh gốc")
        self.seg_frame, self.seg_label = self._image_card(images, "Đối tượng đã phân đoạn")
        self.orig_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.seg_frame.grid(row=0, column=1, sticky="nsew")

    def _image_card(self, parent, title):
        card = ctk.CTkFrame(parent)
        ctk.CTkLabel(card, text=title).pack(anchor="w", padx=12, pady=(10, 4))
        label = ctk.CTkLabel(card, text="Chưa chọn ảnh", width=400, height=400)
        label.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return card, label

    def _build_eval_tab(self):
        tab = self.tab_eval
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", pady=(8, 4))
        self.eval_btn = ctk.CTkButton(
            bar, text="Trích đặc trưng & đánh giá", command=self.start_evaluate
        )
        self.eval_btn.pack(side="left")
        self.rebuild_btn = ctk.CTkButton(
            bar,
            text="Tính lại đặc trưng",
            fg_color="transparent",
            border_width=1,
            command=lambda: self.start_evaluate(force=True),
        )
        self.rebuild_btn.pack(side="left", padx=8)

        self.progress = ctk.CTkProgressBar(tab)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.progress.set(0)

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.metrics_text = ctk.CTkTextbox(body, height=110)
        self.metrics_text.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.metrics_text.insert(
            "1.0",
            "Chưa đánh giá. Nhấn nút phía trên để chạy k-NN trên tập data/ "
            "(80% train / 20% test, có cache sau lần đầu).",
        )
        self.metrics_text.configure(state="disabled")

        self.acc_canvas_host = ctk.CTkFrame(body)
        self.cm_canvas_host = ctk.CTkFrame(body)
        self.acc_canvas_host.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.cm_canvas_host.grid(row=1, column=1, sticky="nsew")
        self.acc_canvas = None
        self.cm_canvas = None

    def _on_k_change(self, value):
        k = max(1, int(round(float(value))))
        self.k_var.set(k)
        self.k_label.configure(text=f"k = {k}")

    def _metric(self):
        return METRIC_LABELS[self.metric_seg.get()]

    def _set_status(self, text):
        self.status.configure(text=text)

    def _set_textbox(self, box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _show_bgr(self, widget, bgr, key):
        pil = bgr_to_pil(bgr)
        img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.photo_refs[key] = img
        widget.configure(image=img, text="")

    def _try_load_cache(self):
        cached = load_feature_cache()
        if cached is None:
            self._set_status(
                "Chưa có cache đặc trưng. Mở tab Đánh giá rồi nhấn Trích đặc trưng & đánh giá."
            )
            return
        X, y, paths = cached
        self._store_dataset(X, y, paths)
        self._set_status(
            f"Đã nạp {len(y)} mẫu từ cache. Có thể nhận dạng hoặc đánh giá lại."
        )

    def _store_dataset(self, X, y, paths):
        self.X, self.y, self.paths = X, y, paths
        self.train_idx, self.test_idx = stratified_split(y, test_ratio=0.2, seed=42)
        self.mean, self.std = standardize_fit(X[self.train_idx])

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh thú cưng",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp"), ("Tất cả", "*.*")],
        )
        if not path:
            return
        try:
            bgr = load_bgr(path)
            img, mask = segment_pet(bgr)
            self.current_path = path
            self._show_bgr(self.orig_label, img, "orig")
            self._show_bgr(self.seg_label, overlay_mask(img, mask), "seg")
            self.result_title.configure(text=os.path.basename(path))
            self._set_textbox(self.neighbor_box, "Nhấn Nhận dạng để dự đoán giống.")
            self._set_status(f"Đã mở {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Lỗi ảnh", str(exc))

    def predict_current(self):
        if not self.current_path:
            messagebox.showinfo("Thiếu ảnh", "Hãy mở một ảnh trước.")
            return
        if self.X is None:
            messagebox.showinfo(
                "Chưa có tập train",
                "Hãy vào tab Đánh giá và chạy trích đặc trưng trước.",
            )
            return
        try:
            vector, img, preview = features_and_preview(self.current_path)
            self._show_bgr(self.orig_label, img, "orig")
            self._show_bgr(self.seg_label, preview, "seg")
            x = standardize_apply(vector[None, :], self.mean, self.std)[0]
            X_train = standardize_apply(self.X[self.train_idx], self.mean, self.std)
            y_train = self.y[self.train_idx]
            k = self.k_var.get()
            pred, dists, labels = knn_predict_one(
                x, X_train, y_train, k=k, metric=self._metric()
            )
            self.result_title.configure(text=DISPLAY_NAMES.get(pred, pred))
            lines = [f"k = {k} · {self.metric_seg.get()}", ""]
            for i, (lab, dist) in enumerate(zip(labels, dists), start=1):
                lines.append(f"{i}. {DISPLAY_NAMES.get(lab, lab):<16}  d = {dist:.3f}")
            self._set_textbox(self.neighbor_box, "\n".join(lines))
            self._set_status(f"Dự đoán: {DISPLAY_NAMES.get(pred, pred)}")
        except Exception as exc:
            messagebox.showerror("Lỗi nhận dạng", str(exc))

    def start_evaluate(self, force=False):
        if self.eval_running:
            return
        self.eval_running = True
        self.eval_btn.configure(state="disabled")
        self.rebuild_btn.configure(state="disabled")
        self._set_status("Đang trích đặc trưng / đánh giá…")
        metric = self._metric()
        active_k = max(1, int(self.k_var.get()))
        threading.Thread(
            target=self._eval_worker,
            args=(force, metric, active_k),
            daemon=True,
        ).start()

    def _eval_worker(self, force, metric, active_k):
        try:
            X, y, paths = load_or_extract(progress=self._progress_from_thread, force=force)
            train_idx, test_idx = stratified_split(y, test_ratio=0.2, seed=42)
            mean, std = standardize_fit(X[train_idx])
            X_train = standardize_apply(X[train_idx], mean, std)
            X_test = standardize_apply(X[test_idx], mean, std)
            y_train, y_test = y[train_idx], y[test_idx]

            k_scores = []
            for k in K_CHOICES:
                preds = knn_predict_batch(X_test, X_train, y_train, k=k, metric=metric)
                k_scores.append(accuracy_score(y_test, preds))

            best_i = int(np.argmax(k_scores))
            best_k = K_CHOICES[best_i]
            preds = knn_predict_batch(
                X_test, X_train, y_train, k=active_k, metric=metric
            )
            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds, SELECTED_BREEDS)
            payload = {
                "X": X,
                "y": y,
                "paths": paths,
                "k_scores": k_scores,
                "best_k": best_k,
                "active_k": active_k,
                "acc": acc,
                "cm": cm,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "metric": metric,
            }
            self.after(0, lambda p=payload: self._eval_done(p))
        except Exception as exc:
            self.after(0, lambda e=exc: self._eval_fail(e))

    def _progress_from_thread(self, i, n, path):
        if i == n or i % 5 == 0:
            self.after(
                0, lambda i=i, n=n, path=path: self._update_progress(i, n, path)
            )

    def _update_progress(self, i, n, path):
        self.progress.set(i / max(n, 1))
        self._set_status(f"Trích đặc trưng {i}/{n}: {os.path.basename(path)}")

    def _eval_done(self, payload):
        self._store_dataset(payload["X"], payload["y"], payload["paths"])
        self.progress.set(1)
        lines = [
            f"Tập: {payload['n_train']} train · {payload['n_test']} test · 5 lớp",
            f"Khoảng cách: {payload['metric']} · k đang chọn = {payload['active_k']}",
            f"Accuracy (k={payload['active_k']}): {payload['acc'] * 100:.2f}%",
            f"k tốt nhất trên test: k={payload['best_k']} "
            f"({max(payload['k_scores']) * 100:.2f}%)",
            "Accuracy theo k: "
            + "  ".join(
                f"k={k}:{s * 100:.1f}%" for k, s in zip(K_CHOICES, payload["k_scores"])
            ),
        ]
        self._set_textbox(self.metrics_text, "\n".join(lines))
        self._draw_accuracy(payload["k_scores"])
        self._draw_confusion(payload["cm"])
        self.eval_running = False
        self.eval_btn.configure(state="normal")
        self.rebuild_btn.configure(state="normal")
        self._set_status("Đánh giá xong.")

    def _eval_fail(self, exc):
        self.eval_running = False
        self.eval_btn.configure(state="normal")
        self.rebuild_btn.configure(state="normal")
        self._set_status("Đánh giá thất bại.")
        messagebox.showerror("Lỗi đánh giá", str(exc))

    def _draw_accuracy(self, scores):
        fig = Figure(figsize=(5.2, 3.6), dpi=100)
        fig.patch.set_facecolor("#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#2b2b2b")
        ax.plot(K_CHOICES, [s * 100 for s in scores], marker="o", color="#5ea1ff")
        ax.set_title("Ảnh hưởng của k", color="white")
        ax.set_xlabel("k", color="white")
        ax.set_ylabel("Accuracy (%)", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        self._embed_fig(self.acc_canvas_host, fig, "acc_canvas")

    def _draw_confusion(self, cm):
        fig = Figure(figsize=(5.2, 3.6), dpi=100)
        fig.patch.set_facecolor("#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#2b2b2b")
        im = ax.imshow(cm, cmap="Blues")
        labels = [DISPLAY_NAMES[c] for c in SELECTED_BREEDS]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=8)
        ax.set_yticklabels(labels, color="white", fontsize=8)
        ax.set_xlabel("Dự đoán", color="white")
        ax.set_ylabel("Thực tế", color="white")
        ax.set_title("Ma trận nhầm lẫn", color="white")
        vmax = max(int(cm.max()), 1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        self._embed_fig(self.cm_canvas_host, fig, "cm_canvas")

    def _embed_fig(self, host, fig, attr):
        old = getattr(self, attr)
        if old is not None:
            old.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=8, pady=8)
        setattr(self, attr, canvas)
