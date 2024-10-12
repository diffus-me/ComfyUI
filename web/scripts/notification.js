import { api } from "./api.js";



class Notification {
  constructor() {
    this.executedNodes = 0;
  }

  onPromptQueued({ detail }) {
    notifier.info("prompt queued");
  }

  onNodeExecuted({ detail }) {
    this.executedNodes++;
  }

  onPromptFinished({ detail }) {
    const consumption = detail.subscription_consumption.credit_consumption;
    const discount = detail.subscription_consumption.discount;
    const charged = Math.ceil(consumption * (1 - discount));
    notifier.info(`prompt finished, used time: <b>${detail.used_time.toFixed(2)}</b>s, credits consumption: <b><strike>${consumption}</strike> ${charged}<b>`);
  }

  onInputCleared({ detail }) {
    this.executedNodes++;
    notifier.info(`input folder cleared: ${detail.user_hash}`);
  }

  onMonitorError({ detail }) {
    upgradeCheck(detail.message)
  }
}

const notification = new Notification();

export function setupNotification() {
  api.addEventListener("promptQueued", (data) => {
    notification.onPromptQueued(data);
  })

  api.addEventListener("executed", (data) => {
    notification.onNodeExecuted(data);
  })

  api.addEventListener("finished", (data) => {
    notification.onPromptFinished(data);
  })

  api.addEventListener("input_cleared", (data) => {
    notification.onInputCleared(data);
  })

  api.addEventListener("monitor_error", (data) => {
    notification.onMonitorError(data);
  })
}
