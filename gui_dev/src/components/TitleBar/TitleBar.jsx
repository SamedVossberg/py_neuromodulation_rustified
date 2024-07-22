import { useWebviewStore } from "@/stores";
import styles from "./TitleBar.module.css";

export const TitleBar = () => {
  const { isWebviewReady, isMaximized, setIsMaximized } = useWebviewStore(
    (state) => ({
      isWebviewReady: state.isWebviewReady,
      isMaximized: state.isMaximized,
      setIsMaximized: state.setIsMaximized,
    })
  );

  const handleMinimizeWindow = () => {
    if (isWebviewReady && window.pywebview && window.pywebview.api) {
      window.pywebview.api.minimize_window();
    }
  };

  const handleMaximizeWindow = () => {
    if (isWebviewReady && window.pywebview && window.pywebview.api) {
      if (isMaximized) {
        window.pywebview.api.restore_window();
        setIsMaximized(false);
        console.log("restoring");
      } else {
        window.pywebview.api.maximize_window();
        setIsMaximized(true);
        console.log("maximizing");
      }
    }
  };

  const handleCloseWindow = () => {
    if (isWebviewReady && window.pywebview && window.pywebview.api) {
      window.pywebview.api.close_window();
    }
  };

  return (
    <div className={styles.titleBar}>
      <a href="https://neuromodulation.github.io/py_neuromodulation/">
        <h1>PyNeuromodulation</h1>
      </a>
      <div className={`${styles.dragArea} pywebview-drag-region`}></div>
      <div className={styles.buttons}>
        <button
          className={styles.buttonMinimize}
          onClick={handleMinimizeWindow}
        >
          -
        </button>
        <button
          className={styles.buttonMaximize}
          onClick={handleMaximizeWindow}
        >
          &#x25A1;
        </button>
        <button className={styles.buttonClose} onClick={handleCloseWindow}>
          x
        </button>
      </div>
    </div>
  );
};
