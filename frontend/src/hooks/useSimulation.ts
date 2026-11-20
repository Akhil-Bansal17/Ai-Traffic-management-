/** Hook for simulation control actions. */

"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { demoSimulator } from "@/lib/demoSimulator";
import { useTrafficStore } from "@/store/trafficStore";
import type { SimulationConfig } from "@/lib/types";

export function useSimulation() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const setStatus = useTrafficStore((s) => s.setStatus);
  const setIsDemoMode = useTrafficStore((s) => s.setIsDemoMode);
  const reset = useTrafficStore((s) => s.reset);

  const startSimulation = async (cfg: Partial<SimulationConfig> = {}) => {
    setLoading(true);
    setError(null);
    try {
      await api.startSimulation(cfg as Record<string, unknown>);
      setIsDemoMode(false);
      setStatus("running");
    } catch (e) {
      console.warn("Backend API unreachable, launching Interactive Demo Mode:", e);
      // Seamlessly switch to Interactive Demo Mode so Vercel users get a live simulation
      demoSimulator.start(cfg);
      setIsDemoMode(true);
      setStatus("running");
    } finally {
      setLoading(false);
    }
  };

  const stopSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      demoSimulator.stop();
      await api.stopSimulation().catch(() => {});
      setIsDemoMode(false);
      setStatus("stopped");
      reset();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to stop simulation");
    } finally {
      setLoading(false);
    }
  };

  const pauseSimulation = async () => {
    try {
      demoSimulator.pause();
      await api.pauseSimulation().catch(() => {});
      setStatus("paused");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to pause");
    }
  };

  const resumeSimulation = async () => {
    try {
      demoSimulator.resume();
      await api.resumeSimulation().catch(() => {});
      setStatus("running");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to resume");
    }
  };

  return {
    startSimulation,
    stopSimulation,
    pauseSimulation,
    resumeSimulation,
    loading,
    error,
  };
}
