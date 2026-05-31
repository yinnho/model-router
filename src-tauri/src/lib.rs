mod api;
mod axum_server;
mod config;
mod convert;
mod proxy;
mod takeover;
mod tray;

use config::load_config;
use tauri::Manager;

pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let result = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // 1. Load config
            let app_config = load_config().map_err(|e| e.to_string())?;

            // 2. Resolve web_dist path
            let web_dist = if cfg!(debug_assertions) {
                // Dev mode: Vite serves frontend, axum doesn't need web_dist
                std::path::PathBuf::from("/dev/null")
            } else {
                // Production: assets are bundled as resources
                app.path()
                    .resource_dir()
                    .map_err(|e| e.to_string())?
                    .join("web")
                    .join("dist")
            };
            log::info!("[Tauri] web_dist path: {}", web_dist.display());

            // 3. Create shared AppState
            let state = config::AppState::new(app_config, web_dist)
                .map_err(|e| e.to_string())?;

            // 4. Start axum server in background thread with its own runtime
            let (tx, rx) = std::sync::mpsc::channel();
            let server_state = state.clone();
            std::thread::spawn(move || {
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        log::error!("[Tauri] failed to create tokio runtime: {}", e);
                        return;
                    }
                };
                let port = rt.block_on(async { axum_server::start(server_state).await });
                let _ = tx.send(port);
                rt.block_on(std::future::pending::<()>()); // Keep runtime alive forever
            });

            // 5. Wait for server to be ready
            let port = rx.recv().map_err(|e| format!("server failed to start: {}", e))?;
            log::info!("[Tauri] axum server ready on port {}", port);

            // 6. Create main window
            // In dev mode, load from Vite dev server (API calls are proxied by Vite)
            // In production, load from axum (which serves bundled static files)
            let window_url = if cfg!(debug_assertions) {
                // cannot fail: well-known URL format
                "http://localhost:5173".parse().unwrap()
            } else {
                // cannot fail: well-formed URL from known port
                format!("http://127.0.0.1:{}", port).parse().unwrap()
            };
            log::info!("[Tauri] opening window at {}", window_url);

            let window = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::External(window_url),
            )
            .title("Model Router")
            .inner_size(960.0, 700.0)
            .min_inner_size(600.0, 400.0)
            .center()
            .build()?;

            // 7. Close-to-tray: intercept close request, hide instead
            let window_clone = window.clone();
            window.on_window_event(move |event| {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window_clone.hide();
                }
            });

            // 8. Set up system tray
            tray::setup_tray(app)?;

            Ok(())
        })
        .run(tauri::generate_context!());

    if let Err(e) = result {
        log::error!("[Tauri] fatal error: {}", e);
        panic!("error while running tauri application: {}", e);
    }
}
