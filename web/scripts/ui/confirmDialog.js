import { $el } from "../ui.js";

export class ComfyConfirmDialog {
    constructor() {
        this.element = $el("div.comfy-modal", { parent: document.body }, [
            $el("div.comfy-modal-content", [$el("p", { $: (p) => (this.textElement = p) }), ...this.createButtons()]),
        ]);
        this.onOkCallback = null;
        this.onCancelCallback = null;
    }

    createButtons() {
        return [
            $el("button", {
                type: "button",
                textContent: "OK",
                onclick: () => this.onOkClicked(),
            }),
            $el("button", {
                type: "button",
                textContent: "Cancel",
                onclick: () => this.onCancelClicked(),
            }),
        ];
    }

    close() {
        this.element.style.display = "none";
        this.onOkCallback = null;
        this.onCancelCallback = null;
    }

    onOkClicked() {
        if (this.onOkCallback) {
            this.onOkCallback();
        }
    }

    onCancelClicked() {
        if (this.onCancelCallback) {
            this.onCancelCallback();
        }
    }

    show(html, onOK, onCancel) {
        if (typeof html === "string") {
            this.textElement.innerHTML = html;
        } else {
            this.textElement.replaceChildren(html);
        }
        this.element.style.display = "flex";
        this.onOkCallback = onOK;
        this.onCancelCallback = onCancel;
    }
}
