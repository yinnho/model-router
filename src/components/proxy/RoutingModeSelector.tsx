/**
 * 路由模式选择器
 *
 * 放置在主界面头部，用于切换路由模式：Off / Opus 4.7 / GPT 5.5 / Auto
 */

import { useRouterConfig, useSetRoutingMode } from "@/lib/query/router";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import type { RoutingMode } from "@/types/proxy";

interface RoutingModeSelectorProps {
  className?: string;
}

const MODES: { mode: RoutingMode; labelKey: string; defaultLabel: string }[] = [
  { mode: "off", labelKey: "router.mode.off", defaultLabel: "Off" },
  { mode: "fixed:opus", labelKey: "router.mode.opus", defaultLabel: "Opus" },
  { mode: "fixed:gpt55", labelKey: "router.mode.gpt", defaultLabel: "GPT" },
  { mode: "auto", labelKey: "router.mode.auto", defaultLabel: "Auto" },
];

export function RoutingModeSelector({ className }: RoutingModeSelectorProps) {
  const { t } = useTranslation();
  const { data: config } = useRouterConfig();
  const setMode = useSetRoutingMode();

  const currentMode = config?.mode ?? "off";

  return (
    <div
      className={cn(
        "flex items-center gap-0.5 h-8 rounded-lg bg-muted/50 p-0.5",
        className,
      )}
    >
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
  );
}
