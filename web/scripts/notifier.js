import { api } from "./api.js";



class Notifier {
  notifierGlobalOptions = {
    position: "bottom-right",
    icons: { enabled: false },
    minDurations: {
      async: 30,
      "async-block": 30,
    },
  };

  constructor() {
    this.awn = new AWN(this.notifierGlobalOptions);
    this.executedNodes = 0;
  }

  onPromptQueued({ detail }) {
    this.awn.info("prompt queued");
  }

  onNodeExecuted({ detail }) {
    this.executedNodes++;
  }

  onPromptFinished({ detail }) {
    const consumption = detail.subscription_consumption.credit_consumption;
    const discount = detail.subscription_consumption.discount;
    const charged = Math.ceil(consumption * (1 - discount));
    this.awn.info(`prompt finished, used time: <b>${detail.used_time.toFixed(2)}</b>s, credits consumption: <b><strike>${consumption}</strike> ${charged}<b>`);
  }

  onInputCleared({ detail }) {
    this.executedNodes++;
    this.awn.info(`input folder cleared: ${detail.user_hash}`);
  }
}

const notifier = new Notifier();

export function setupNotifier() {
  api.addEventListener("promptQueued", (data) => {
    notifier.onPromptQueued(data);
  })

  api.addEventListener("executed", (data) => {
    notifier.onNodeExecuted(data);
  })

  api.addEventListener("finished", (data) => {
    notifier.onPromptFinished(data);
  })

  api.addEventListener("input_cleared", (data) => {
    notifier.onInputCleared(data);
  })
}
