import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { routerApi } from "@/lib/api/router";
import type { RouterConfig, RoutingMode } from "@/types/proxy";

const ROUTER_CONFIG_KEY = ["routerConfig"] as const;
const ROUTER_STATUS_KEY = (appType: string) =>
  ["routerStatus", appType] as const;

export function useRouterConfig() {
  return useQuery({
    queryKey: ROUTER_CONFIG_KEY,
    queryFn: () => routerApi.getConfig(),
    staleTime: 30_000,
  });
}

export function useRouterStatus(appType: string) {
  return useQuery({
    queryKey: ROUTER_STATUS_KEY(appType),
    queryFn: () => routerApi.checkStatus(appType),
    staleTime: 10_000,
    enabled: !!appType,
  });
}

export function useUpdateRouterConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (config: RouterConfig) => routerApi.updateConfig(config),
    onMutate: async (newConfig) => {
      await queryClient.cancelQueries({ queryKey: ROUTER_CONFIG_KEY });
      const previous = queryClient.getQueryData<RouterConfig>(ROUTER_CONFIG_KEY);
      queryClient.setQueryData<RouterConfig>(ROUTER_CONFIG_KEY, newConfig);
      return { previous };
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(ROUTER_CONFIG_KEY, context.previous);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ROUTER_CONFIG_KEY });
      queryClient.invalidateQueries({ queryKey: ["proxyStatus"] });
      queryClient.invalidateQueries({ queryKey: ["routerStatus"] });
    },
  });
}

export function useSetRoutingMode() {
  const queryClient = useQueryClient();
  const { data: config } = useRouterConfig();

  return useMutation({
    mutationFn: (mode: RoutingMode) => {
      const current = config ?? {
        mode: "off" as RoutingMode,
        classifierModel: "claude-haiku-4-5",
        cacheTtlSeconds: 300,
      };
      return routerApi.updateConfig({ ...current, mode });
    },
    onMutate: async (mode) => {
      await queryClient.cancelQueries({ queryKey: ROUTER_CONFIG_KEY });
      const previous = queryClient.getQueryData<RouterConfig>(ROUTER_CONFIG_KEY);
      if (previous) {
        queryClient.setQueryData<RouterConfig>(ROUTER_CONFIG_KEY, {
          ...previous,
          mode,
        });
      }
      return { previous };
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(ROUTER_CONFIG_KEY, context.previous);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ROUTER_CONFIG_KEY });
      queryClient.invalidateQueries({ queryKey: ["proxyStatus"] });
      queryClient.invalidateQueries({ queryKey: ["routerStatus"] });
    },
  });
}
