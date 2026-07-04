use crate::profile::RunMetrics;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use sysinfo::{Pid, System};
use tokio::io::AsyncWriteExt;
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TelemetrySample {
    pub timestamp: DateTime<Utc>,
    pub gpu_index: Option<u32>,
    pub vram_used_mib: Option<u64>,
    pub vram_free_mib: Option<u64>,
    pub gpu_util_pct: Option<u64>,
    pub ram_available_mib: Option<u64>,
    pub swap_used_mib: Option<u64>,
    pub process_rss_mib: Option<u64>,
    pub process_cpu_pct: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TelemetrySummary {
    pub peak_vram_mib: Option<u64>,
    pub min_free_vram_mib: Option<u64>,
    pub gpu_util_avg_pct: Option<f64>,
    pub gpu_util_max_pct: Option<u64>,
    pub ram_available_min_mib: Option<u64>,
    pub swap_start_mib: Option<u64>,
    pub swap_end_mib: Option<u64>,
    pub swap_delta_mib: Option<i64>,
    pub process_rss_peak_mib: Option<u64>,
    pub cpu_util_avg_pct: Option<f64>,
    pub sample_count: u64,
}

pub struct TelemetrySampler {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<TelemetrySummary>,
}

impl TelemetrySampler {
    pub fn start(
        pid: u32,
        output_path: impl AsRef<Path>,
        gpu_index: Option<u32>,
        interval: Duration,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_task = Arc::clone(&stop);
        let output_path = output_path.as_ref().to_path_buf();
        let handle = tokio::spawn(async move {
            sample_loop(pid, output_path, gpu_index, interval, stop_for_task)
                .await
                .unwrap_or_default()
        });
        Self { stop, handle }
    }

    pub async fn stop(self) -> TelemetrySummary {
        self.stop.store(true, Ordering::SeqCst);
        self.handle.await.unwrap_or_default()
    }
}

impl From<TelemetrySummary> for RunMetrics {
    fn from(summary: TelemetrySummary) -> Self {
        Self {
            peak_vram_mib: summary.peak_vram_mib,
            min_free_vram_mib: summary.min_free_vram_mib,
            gpu_util_avg_pct: summary.gpu_util_avg_pct,
            gpu_util_max_pct: summary.gpu_util_max_pct,
            ram_available_min_mib: summary.ram_available_min_mib,
            swap_delta_mib: summary.swap_delta_mib,
            process_rss_peak_mib: summary.process_rss_peak_mib,
            cpu_util_avg_pct: summary.cpu_util_avg_pct,
            ..RunMetrics::default()
        }
    }
}

pub fn collect_once(pid: Option<u32>, gpu_index: Option<u32>) -> TelemetrySample {
    let mut sample = TelemetrySample {
        timestamp: Utc::now(),
        gpu_index,
        ..TelemetrySample::default()
    };
    if let Some(gpu) = query_nvidia_smi(gpu_index) {
        sample.vram_used_mib = gpu.vram_used_mib;
        sample.vram_free_mib = gpu.vram_free_mib;
        sample.gpu_util_pct = gpu.gpu_util_pct;
    }
    if let Some(memory) = read_meminfo() {
        sample.ram_available_mib = memory.ram_available_mib;
        sample.swap_used_mib = memory.swap_used_mib;
    }
    if let Some(pid) = pid {
        let process = read_process(pid);
        sample.process_rss_mib = process.rss_mib;
        sample.process_cpu_pct = process.cpu_pct;
    }
    sample
}

async fn sample_loop(
    pid: u32,
    output_path: PathBuf,
    gpu_index: Option<u32>,
    interval: Duration,
    stop: Arc<AtomicBool>,
) -> Result<TelemetrySummary> {
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&output_path)
        .await
        .with_context(|| format!("open {}", output_path.display()))?;
    let mut summary = TelemetryAccumulator::default();

    while !stop.load(Ordering::SeqCst) {
        let sample = collect_once(Some(pid), gpu_index);
        let line = serde_json::to_string(&sample)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        summary.push(&sample);
        tokio::time::sleep(interval).await;
    }

    Ok(summary.finish())
}

#[derive(Default)]
struct TelemetryAccumulator {
    peak_vram_mib: Option<u64>,
    min_free_vram_mib: Option<u64>,
    gpu_util_total: u64,
    gpu_util_samples: u64,
    gpu_util_max_pct: Option<u64>,
    ram_available_min_mib: Option<u64>,
    swap_start_mib: Option<u64>,
    swap_end_mib: Option<u64>,
    process_rss_peak_mib: Option<u64>,
    cpu_util_total: f64,
    cpu_util_samples: u64,
    sample_count: u64,
}

impl TelemetryAccumulator {
    fn push(&mut self, sample: &TelemetrySample) {
        self.sample_count += 1;
        if let Some(value) = sample.vram_used_mib {
            self.peak_vram_mib = Some(self.peak_vram_mib.map_or(value, |peak| peak.max(value)));
        }
        if let Some(value) = sample.vram_free_mib {
            self.min_free_vram_mib =
                Some(self.min_free_vram_mib.map_or(value, |min| min.min(value)));
        }
        if let Some(value) = sample.gpu_util_pct {
            self.gpu_util_total += value;
            self.gpu_util_samples += 1;
            self.gpu_util_max_pct = Some(self.gpu_util_max_pct.map_or(value, |max| max.max(value)));
        }
        if let Some(value) = sample.ram_available_mib {
            self.ram_available_min_mib = Some(
                self.ram_available_min_mib
                    .map_or(value, |min| min.min(value)),
            );
        }
        if let Some(value) = sample.swap_used_mib {
            if self.swap_start_mib.is_none() {
                self.swap_start_mib = Some(value);
            }
            self.swap_end_mib = Some(value);
        }
        if let Some(value) = sample.process_rss_mib {
            self.process_rss_peak_mib = Some(
                self.process_rss_peak_mib
                    .map_or(value, |peak| peak.max(value)),
            );
        }
        if let Some(value) = sample.process_cpu_pct {
            self.cpu_util_total += f64::from(value);
            self.cpu_util_samples += 1;
        }
    }

    fn finish(self) -> TelemetrySummary {
        TelemetrySummary {
            peak_vram_mib: self.peak_vram_mib,
            min_free_vram_mib: self.min_free_vram_mib,
            gpu_util_avg_pct: average(self.gpu_util_total as f64, self.gpu_util_samples),
            gpu_util_max_pct: self.gpu_util_max_pct,
            ram_available_min_mib: self.ram_available_min_mib,
            swap_start_mib: self.swap_start_mib,
            swap_end_mib: self.swap_end_mib,
            swap_delta_mib: self
                .swap_start_mib
                .zip(self.swap_end_mib)
                .map(|(start, end)| end as i64 - start as i64),
            process_rss_peak_mib: self.process_rss_peak_mib,
            cpu_util_avg_pct: average(self.cpu_util_total, self.cpu_util_samples),
            sample_count: self.sample_count,
        }
    }
}

fn average(total: f64, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

#[derive(Debug)]
struct GpuSample {
    vram_used_mib: Option<u64>,
    vram_free_mib: Option<u64>,
    gpu_util_pct: Option<u64>,
}

fn query_nvidia_smi(gpu_index: Option<u32>) -> Option<GpuSample> {
    let mut command = std::process::Command::new("nvidia-smi");
    command.args([
        "--query-gpu=memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]);
    if let Some(index) = gpu_index {
        command.args(["-i", &index.to_string()]);
    }
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?;
    let mut parts = line.split(',').map(|part| part.trim().parse::<u64>().ok());
    Some(GpuSample {
        vram_used_mib: parts.next().flatten(),
        vram_free_mib: parts.next().flatten(),
        gpu_util_pct: parts.next().flatten(),
    })
}

#[derive(Debug)]
struct MemorySample {
    ram_available_mib: Option<u64>,
    swap_used_mib: Option<u64>,
}

fn read_meminfo() -> Option<MemorySample> {
    let data = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut mem_available_kib = None;
    let mut swap_total_kib = None;
    let mut swap_free_kib = None;

    for line in data.lines() {
        if let Some(value) = parse_meminfo_line(line, "MemAvailable:") {
            mem_available_kib = Some(value);
        } else if let Some(value) = parse_meminfo_line(line, "SwapTotal:") {
            swap_total_kib = Some(value);
        } else if let Some(value) = parse_meminfo_line(line, "SwapFree:") {
            swap_free_kib = Some(value);
        }
    }

    Some(MemorySample {
        ram_available_mib: mem_available_kib.map(|value| value / 1024),
        swap_used_mib: swap_total_kib
            .zip(swap_free_kib)
            .map(|(total, free)| total.saturating_sub(free) / 1024),
    })
}

fn parse_meminfo_line(line: &str, key: &str) -> Option<u64> {
    line.strip_prefix(key)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[derive(Debug)]
struct ProcessSample {
    rss_mib: Option<u64>,
    cpu_pct: Option<f32>,
}

fn read_process(pid: u32) -> ProcessSample {
    let mut system = System::new();
    let pid = Pid::from_u32(pid);
    system.refresh_process(pid);
    if let Some(process) = system.process(pid) {
        return ProcessSample {
            rss_mib: Some(process.memory() / 1024 / 1024),
            cpu_pct: Some(process.cpu_usage()),
        };
    }
    ProcessSample {
        rss_mib: read_proc_status_rss(pid.as_u32()),
        cpu_pct: None,
    }
}

fn read_proc_status_rss(pid: u32) -> Option<u64> {
    let data = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in data.lines() {
        if let Some(kib) = parse_meminfo_line(line, "VmRSS:") {
            return Some(kib / 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_tracks_min_max_and_delta() {
        let mut acc = TelemetryAccumulator::default();
        acc.push(&TelemetrySample {
            timestamp: Utc::now(),
            vram_used_mib: Some(100),
            vram_free_mib: Some(1000),
            gpu_util_pct: Some(10),
            ram_available_mib: Some(8000),
            swap_used_mib: Some(0),
            process_rss_mib: Some(50),
            process_cpu_pct: Some(5.0),
            gpu_index: Some(0),
        });
        acc.push(&TelemetrySample {
            timestamp: Utc::now(),
            vram_used_mib: Some(300),
            vram_free_mib: Some(700),
            gpu_util_pct: Some(30),
            ram_available_mib: Some(7000),
            swap_used_mib: Some(128),
            process_rss_mib: Some(75),
            process_cpu_pct: Some(15.0),
            gpu_index: Some(0),
        });
        let summary = acc.finish();
        assert_eq!(summary.peak_vram_mib, Some(300));
        assert_eq!(summary.min_free_vram_mib, Some(700));
        assert_eq!(summary.gpu_util_avg_pct, Some(20.0));
        assert_eq!(summary.swap_delta_mib, Some(128));
        assert_eq!(summary.process_rss_peak_mib, Some(75));
    }
}
