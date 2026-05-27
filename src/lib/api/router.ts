import { invoke } from "@tauri-apps/api/core";
import type { RouterConfig } from "@/types/proxy";

export const routerApi = {
  async getConfig(): Promise<RouterConfig> {
    return invoke("get_router_config");
  },

  async updateConfig(config: RouterConfig): Promise<void> {
    return invoke("update_router_config", { config });
  },
};
