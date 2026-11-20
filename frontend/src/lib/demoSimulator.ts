/**
 * Interactive client-side demo simulator.
 * Automatically runs when the frontend is hosted on Vercel or when the local
 * Python/SUMO backend is not directly reachable over HTTPS (mixed content prevention).
 */

import { useTrafficStore } from "@/store/trafficStore";
import type {
  IntersectionState,
  MetricsSnapshot,
  SimulationConfig,
  TickData,
  VehicleState,
} from "./types";

interface SimVehicle {
  id: string;
  type: "car" | "truck" | "bus" | "motorcycle";
  edge: string;
  lane: number;
  dir: "E" | "W" | "N" | "S";
  pos: number;
  speed: number;
  targetSpeed: number;
  wait: number;
}

const NODES = ["A1", "B1", "C1", "A0", "B0", "C0"] as const;

// World positions
const NODE_COORDS: Record<string, { x: number; y: number }> = {
  A1: { x: 500, y: 1000 },
  B1: { x: 1200, y: 1000 },
  C1: { x: 1900, y: 1000 },
  A0: { x: 500, y: 500 },
  B0: { x: 1200, y: 500 },
  C0: { x: 1900, y: 500 },
};

class DemoSimulator {
  private timer: NodeJS.Timeout | null = null;
  private isPaused = false;
  private tick = 0;
  private simTime = 0;
  private vehicles: SimVehicle[] = [];
  private phaseTimer = 0;
  private currentPhases: Record<string, number> = {};
  private config: Partial<SimulationConfig> = {};
  private totalCompleted = 0;

  constructor() {
    this.resetState();
  }

  private resetState() {
    this.tick = 0;
    this.simTime = 0;
    this.phaseTimer = 0;
    this.totalCompleted = 0;
    for (const node of NODES) {
      this.currentPhases[node] = 0;
    }
    this.initVehicles();
  }

  private initVehicles() {
    const edges: { edge: string; dir: "E" | "W" | "N" | "S"; minPos: number; maxPos: number; lanes: number }[] = [
      { edge: "A1B1", dir: "E", minPos: 520, maxPos: 1180, lanes: 3 },
      { edge: "B1A1", dir: "W", minPos: 520, maxPos: 1180, lanes: 3 },
      { edge: "B1C1", dir: "E", minPos: 1220, maxPos: 1880, lanes: 3 },
      { edge: "C1B1", dir: "W", minPos: 1220, maxPos: 1880, lanes: 3 },
      { edge: "A0B0", dir: "E", minPos: 520, maxPos: 1180, lanes: 3 },
      { edge: "B0A0", dir: "W", minPos: 520, maxPos: 1180, lanes: 3 },
      { edge: "B0C0", dir: "E", minPos: 1220, maxPos: 1880, lanes: 3 },
      { edge: "C0B0", dir: "W", minPos: 1220, maxPos: 1880, lanes: 3 },
      { edge: "A0A1", dir: "N", minPos: 520, maxPos: 980, lanes: 2 },
      { edge: "A1A0", dir: "S", minPos: 520, maxPos: 980, lanes: 2 },
      { edge: "B0B1", dir: "N", minPos: 520, maxPos: 980, lanes: 2 },
      { edge: "B1B0", dir: "S", minPos: 520, maxPos: 980, lanes: 2 },
      { edge: "C0C1", dir: "N", minPos: 520, maxPos: 980, lanes: 2 },
      { edge: "C1C0", dir: "S", minPos: 520, maxPos: 980, lanes: 2 },
    ];

    this.vehicles = [];
    let id = 1;
    const types: ("car" | "truck" | "bus" | "motorcycle")[] = ["car", "car", "car", "truck", "bus", "motorcycle"];

    for (const e of edges) {
      const count = 2 + Math.floor(Math.random() * 3);
      for (let i = 0; i < count; i++) {
        const lane = Math.floor(Math.random() * e.lanes);
        const pos = e.minPos + Math.random() * (e.maxPos - e.minPos);
        this.vehicles.push({
          id: `veh_${id++}`,
          type: types[Math.floor(Math.random() * types.length)],
          edge: e.edge,
          lane,
          dir: e.dir,
          pos,
          speed: 8 + Math.random() * 6,
          targetSpeed: 10 + Math.random() * 5,
          wait: 0,
        });
      }
    }
  }

  start(config: Partial<SimulationConfig> = {}) {
    this.stop();
    this.config = config;
    this.resetState();
    this.isPaused = false;

    // Run at 10 ticks per second (100ms interval)
    this.timer = setInterval(() => {
      if (!this.isPaused) {
        this.step();
      }
    }, 100);
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.isPaused = false;
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    this.isPaused = false;
  }

  isRunning() {
    return this.timer !== null;
  }

  private step() {
    this.tick += 1;
    this.simTime += 0.1;
    this.phaseTimer += 0.1;

    // Phase cycle: 0 = N green, 1 = E green, 2 = S green, 3 = W green
    const phaseDurations = [12, 28, 12, 28]; // E-W arterials get more green
    const currentPhaseIndex = Math.floor((this.phaseTimer % 80) / 20);

    for (const node of NODES) {
      this.currentPhases[node] = currentPhaseIndex;
    }

    // Update vehicle positions and queues
    const queues: Record<string, { N: number; S: number; E: number; W: number }> = {};
    for (const node of NODES) {
      queues[node] = { N: 0, S: 0, E: 0, W: 0 };
    }

    let haltingCount = 0;
    let totalDelay = 0;

    for (const v of this.vehicles) {
      // Determine if vehicle is approaching an intersection
      const isRed = (v.dir === "N" && currentPhaseIndex !== 0) ||
                    (v.dir === "E" && currentPhaseIndex !== 1) ||
                    (v.dir === "S" && currentPhaseIndex !== 2) ||
                    (v.dir === "W" && currentPhaseIndex !== 3);

      // Simple threshold near stop line
      let nearStop = false;
      let targetNode = "";

      if (v.edge === "A1B1" && v.pos > 1100) { nearStop = true; targetNode = "B1"; }
      else if (v.edge === "B1A1" && v.pos < 600) { nearStop = true; targetNode = "A1"; }
      else if (v.edge === "B1C1" && v.pos > 1800) { nearStop = true; targetNode = "C1"; }
      else if (v.edge === "C1B1" && v.pos < 1300) { nearStop = true; targetNode = "B1"; }
      else if (v.edge === "A0B0" && v.pos > 1100) { nearStop = true; targetNode = "B0"; }
      else if (v.edge === "B0A0" && v.pos < 600) { nearStop = true; targetNode = "A0"; }
      else if (v.edge === "B0C0" && v.pos > 1800) { nearStop = true; targetNode = "C0"; }
      else if (v.edge === "C0B0" && v.pos < 1300) { nearStop = true; targetNode = "B0"; }
      else if (v.edge === "A0A1" && v.pos > 920) { nearStop = true; targetNode = "A1"; }
      else if (v.edge === "A1A0" && v.pos < 580) { nearStop = true; targetNode = "A0"; }
      else if (v.edge === "B0B1" && v.pos > 920) { nearStop = true; targetNode = "B1"; }
      else if (v.edge === "B1B0" && v.pos < 580) { nearStop = true; targetNode = "B0"; }
      else if (v.edge === "C0C1" && v.pos > 920) { nearStop = true; targetNode = "C1"; }
      else if (v.edge === "C1C0" && v.pos < 580) { nearStop = true; targetNode = "C0"; }

      if (nearStop && isRed) {
        v.speed = Math.max(0, v.speed - 2);
        v.wait += 0.1;
        haltingCount++;
        if (targetNode && queues[targetNode]) {
          queues[targetNode][v.dir]++;
        }
      } else {
        v.speed = Math.min(v.targetSpeed, v.speed + 1.5);
      }

      totalDelay += v.wait;

      // Move vehicle
      const stepDist = (v.speed * 0.1) * 5; // scaled visual speed
      if (v.dir === "E" || v.dir === "N") {
        v.pos += stepDist;
      } else {
        v.pos -= stepDist;
      }

      // Wrap around edges
      if (v.dir === "E" && v.pos > 2000) { v.pos = 500; this.totalCompleted++; }
      if (v.dir === "W" && v.pos < 450) { v.pos = 1950; this.totalCompleted++; }
      if (v.dir === "N" && v.pos > 1100) { v.pos = 450; this.totalCompleted++; }
      if (v.dir === "S" && v.pos < 450) { v.pos = 1100; this.totalCompleted++; }
    }

    // Build Intersections State
    const phaseNames = ["N_GREEN", "E_GREEN", "S_GREEN", "W_GREEN"];
    const phaseStates = ["GGGrrrrrrrrr", "rrrGGGrrrrrr", "rrrrrrGGGrrr", "rrrrrrrrrGGG"];

    const intersections: IntersectionState[] = NODES.map((node) => {
      const q = queues[node];
      const count = q.N + q.S + q.E + q.W;
      const phaseIdx = this.currentPhases[node];
      return {
        id: node,
        signal_state: phaseStates[phaseIdx],
        phase_index: phaseIdx,
        phase_remaining_s: Math.max(1, 20 - Math.floor(this.phaseTimer % 20)),
        queue_lengths: q,
        vehicle_count: count,
        avg_wait_s: count > 0 ? (count * 4.2) : 0,
      };
    });

    // Build Vehicles State
    const vehicleStates: VehicleState[] = this.vehicles.map((v) => {
      let x = 0;
      let y = 0;

      if (v.edge === "A1B1" || v.edge === "B1C1") {
        x = v.pos;
        y = 1000;
      } else if (v.edge === "B1A1" || v.edge === "C1B1") {
        x = v.pos;
        y = 1000;
      } else if (v.edge === "A0B0" || v.edge === "B0C0") {
        x = v.pos;
        y = 500;
      } else if (v.edge === "B0A0" || v.edge === "C0B0") {
        x = v.pos;
        y = 500;
      } else if (v.edge.startsWith("A")) {
        x = 500;
        y = v.pos;
      } else if (v.edge.startsWith("B")) {
        x = 1200;
        y = v.pos;
      } else {
        x = 1900;
        y = v.pos;
      }

      return {
        id: v.id,
        type: v.type,
        x,
        y,
        speed: v.speed,
        lane_id: `${v.edge}_${v.lane}`,
        accumulated_wait_s: Math.round(v.wait * 10) / 10,
      };
    });

    // Build Metrics
    const policyIsActuated = (this.config.policy_type || "actuated") === "actuated";
    const baseDelay = policyIsActuated ? 16.5 : 32.0;
    const avgDelay = Math.max(5, baseDelay + Math.sin(this.simTime / 5) * 3);

    const metrics: MetricsSnapshot = {
      total_vehicles: this.vehicles.length + Math.floor(this.simTime * 0.5),
      total_completed: this.totalCompleted,
      avg_delay_s: Math.round(avgDelay * 10) / 10,
      throughput_veh_per_min: Math.round((42 + (policyIsActuated ? 12 : 0) + Math.sin(this.simTime / 8) * 4) * 10) / 10,
      total_halting: haltingCount,
      avg_control_delay_s: Math.round((avgDelay * 0.75) * 10) / 10,
      control_delay_samples: this.vehicles.length * 5,
      avg_trip_time_s: Math.round((avgDelay + 45) * 10) / 10,
      completed_trips: this.totalCompleted,
    };

    const tickData: TickData = {
      tick: this.tick,
      sim_time: Math.round(this.simTime * 10) / 10,
      intersections,
      vehicles: vehicleStates,
      metrics,
      emergency: { active: [] },
      camera_frame_url: null,
    };

    // Feed tick data into store
    useTrafficStore.getState().updateFromTick(tickData);
  }
}

export const demoSimulator = new DemoSimulator();
