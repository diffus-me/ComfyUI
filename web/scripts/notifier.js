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

  onInputCleared({ detail }) {
    this.executedNodes++;
    this.awn.info(`input folder cleared.`);
  }
}

const notifier = new Notifier();

export function setupNotifier() {
  api.addEventListener("input_cleared", (data) => {
    notifier.onInputCleared(data);
  })
}
