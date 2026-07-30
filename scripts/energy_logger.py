import time, json, platform
from dataclasses import dataclass, field

# Update cpu_laptop to match your exact CPU's rated base TDP.
REFERENCE_TDP_WATTS = {
    "cpu_laptop": 45,
    "cpu_desktop": 65,
    "colab_cpu": 25,
}
DEFAULT_PUE = 1.0  # 1.0 = no datacenter overhead, correct for your own laptop


@dataclass
class EnergyLogger:
    device_name: str
    device_tdp_watts: float
    pue: float = DEFAULT_PUE
    _start_time: float = field(default=None, init=False, repr=False)
    _end_time: float = field(default=None, init=False, repr=False)
    checkpoints: list = field(default_factory=list, init=False)

    def start(self):
        self._start_time = time.time()
        self.checkpoints.append({"event": "start", "t": self._start_time})
        print(f"[EnergyLogger] Started on {self.device_name} (TDP={self.device_tdp_watts}W, PUE={self.pue})")

    def checkpoint(self, label):
        self.checkpoints.append({"event": label, "t": time.time()})

    def stop(self):
        self._end_time = time.time()
        self.checkpoints.append({"event": "stop", "t": self._end_time})

    @property
    def elapsed_hours(self):
        if self._start_time is None:
            return 0.0
        return ((self._end_time or time.time()) - self._start_time) / 3600.0

    @property
    def energy_kwh(self):
        return (self.device_tdp_watts * self.elapsed_hours * self.pue) / 1000.0

    def report(self):
        print(f"\n[EnergyLogger] {self.device_name}: {self.elapsed_hours:.4f}h, {self.energy_kwh:.6f} kWh")
        return {
            "device_name": self.device_name,
            "elapsed_hours": self.elapsed_hours,
            "device_tdp_watts": self.device_tdp_watts,
            "pue": self.pue,
            "energy_kwh": self.energy_kwh,
            "platform": platform.platform(),
        }

    def save(self, path):
        data = self.report()
        data["checkpoints"] = self.checkpoints
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
