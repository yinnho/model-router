/**
 * 路由模式选择器
 *
 * 放置在主界面头部，用于切换路由模式：Off / Opus / GPT / Auto
 * 当选中模式没有匹配的 provider 时显示警告
 */

import { useRouterConfig, useRouterStatus, useSetRoutingMode } from "@/lib/query/router";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import { AlertTriangle } from "lucide-react";
import type { RoutingMode } from "@/types/proxy";
import type { AppId } from "@/lib/api";

interface RoutingModeSelectorProps {
  className?: string;
  activeApp: AppId;
}

const MODES: { mode: RoutingMode; labelKey: string; defaultLabel: string }[] = [
  { mode: "off", labelKey: "router.mode.off", defaultLabel: "Off" },
  { mode: "fixed:opus", labelKey: "router.mode.opus", defaultLabel: "Opus" },
  { mode: "fixed:gpt55", labelKey: "router.mode.gpt", defaultLabel: "GPT" },
  { mode: "auto", labelKey: "router.mode.auto", defaultLabel: "Auto" },
];

export function RoutingModeSelector({ className, activeApp }: RoutingModeSelectorProps) {
  const { t } = useTranslation();
  const { data: config } = useRouterConfig();
  const { data: status } = useRouterStatus(activeApp);
  const setMode = useSetRoutingMode();

  const currentMode = config?.mode ?? "off";
  const hasMatch = status?.hasMatchingProvider !== false;

  return (
    <div className={cn("flex items-center gap-1", className)}>
      <div className="flex items-center gap-0.5 h-8 rounded-lg bg-muted/50 p-0.5">
        {MODES.map(({ mode, labelKey, defaultLabel }) => {
          const isActive = currentMode === mode;
          return (
            <button
              key={mode}
              onClick={() => {
                if (!isActive) setMode.mutate(mode);
              }}
              className={cn(
                "px-2 h-7 rounded-md text-xs font-medium transition-all",
                isActive
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
              title={t(labelKey, { defaultValue: defaultLabel })}
            >
              {t(labelKey, { defaultValue: defaultLabel })}
            </button>
          );
        })}
      </div>
      {currentMode !== "off" && !hasMatch && (
        <span
          title={t("router.noProvider", {
            defaultValue: "No matching provider for this routing mode",
          })}
        >
          <AlertTriangle className="h-4 w-4 text-amber-500" />
        </span>
      )}
    </div>
  );
}
