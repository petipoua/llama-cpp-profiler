use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvironmentSnapshot {
    pub profiler_version: String,
    pub os: String,
    pub cpu_brand: Option<String>,
    pub cpu_cores: Option<usize>,
    pub ram_total_mib: Option<u64>,
    pub swap_total_mib: Option<u64>,
    pub gpu_backend: String,
    pub gpus: Vec<GpuInfo>,
    pub llama_server: ServerInfo,
}

impl Default for EnvironmentSnapshot {
    fn default() -> Self {
        Self {
            profiler_version: env!("CARGO_PKG_VERSION").to_string(),
            os: std::env::consts::OS.to_string(),
            cpu_brand: None,
            cpu_cores: None,
            ram_total_mib: None,
            swap_total_mib: None,
            gpu_backend: "unknown".to_string(),
            gpus: Vec::new(),
            llama_server: ServerInfo::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuInfo {
    pub index: u32,
    pub name: Option<String>,
    pub total_vram_mib: Option<u64>,
    pub driver_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerInfo {
    pub executable: String,
    pub resolved_path: Option<PathBuf>,
    pub version: Option<String>,
    pub help_hash: Option<String>,
    pub usable: bool,
    pub error: Option<String>,
}

impl Default for ServerInfo {
    fn default() -> Self {
        Self {
            executable: "llama-server".to_string(),
            resolved_path: None,
            version: None,
            help_hash: None,
            usable: false,
            error: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Compatibility {
    Current,
    #[default]
    LegacyMissingSnapshot,
    ServerChanged,
    HardwareChanged,
}

impl Compatibility {
    pub fn is_current(self) -> bool {
        matches!(self, Self::Current)
    }

    pub fn reason(self) -> &'static str {
        match self {
            Self::Current => "current environment",
            Self::LegacyMissingSnapshot => "legacy run is missing environment snapshot",
            Self::ServerChanged => "llama-server executable or help output changed",
            Self::HardwareChanged => "hardware or driver changed",
        }
    }
}

pub fn capture_environment(executable: &str, help: Option<&str>) -> EnvironmentSnapshot {
    let mut system = sysinfo::System::new_all();
    system.refresh_all();
    let (ram_total_mib, swap_total_mib) = read_memory_totals().unwrap_or((
        Some(system.total_memory() / 1024 / 1024),
        Some(system.total_swap() / 1024 / 1024),
    ));
    let server = server_info(executable, help);
    let gpus = query_nvidia_gpus();
    let cpu_brand = non_empty(system.global_cpu_info().brand()).or_else(read_cpu_brand);
    EnvironmentSnapshot {
        profiler_version: env!("CARGO_PKG_VERSION").to_string(),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        cpu_brand,
        cpu_cores: std::thread::available_parallelism().ok().map(usize::from),
        ram_total_mib,
        swap_total_mib,
        gpu_backend: if gpus.is_empty() {
            "unknown".to_string()
        } else {
            "nvidia".to_string()
        },
        gpus,
        llama_server: server,
    }
}

fn non_empty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn read_cpu_brand() -> Option<String> {
    let data = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    data.lines()
        .find_map(|line| line.strip_prefix("model name"))
        .and_then(|line| line.split_once(':').map(|(_, value)| value.trim()))
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub fn compare_environment(
    run: Option<&EnvironmentSnapshot>,
    current: &EnvironmentSnapshot,
) -> Compatibility {
    let Some(run) = run else {
        return Compatibility::LegacyMissingSnapshot;
    };
    if run.llama_server.executable != current.llama_server.executable
        || run.llama_server.resolved_path != current.llama_server.resolved_path
        || run.llama_server.help_hash != current.llama_server.help_hash
    {
        return Compatibility::ServerChanged;
    }
    if run.os != current.os
        || run.cpu_brand != current.cpu_brand
        || run.cpu_cores != current.cpu_cores
        || run.ram_total_mib != current.ram_total_mib
        || run.swap_total_mib != current.swap_total_mib
        || run.gpu_backend != current.gpu_backend
        || run.gpus != current.gpus
    {
        return Compatibility::HardwareChanged;
    }
    Compatibility::Current
}

fn server_info(executable: &str, help: Option<&str>) -> ServerInfo {
    let resolved_path = resolve_executable(executable);
    let version = std::process::Command::new(executable)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
            .trim()
            .to_string()
        })
        .filter(|value| !value.is_empty());
    ServerInfo {
        executable: executable.to_string(),
        resolved_path,
        version,
        help_hash: help.map(stable_hash),
        usable: help.is_some(),
        error: help
            .is_none()
            .then(|| "llama-server help was unavailable".to_string()),
    }
}

fn resolve_executable(executable: &str) -> Option<PathBuf> {
    let path = PathBuf::from(executable);
    if path.components().count() > 1 {
        return std::fs::canonicalize(path).ok();
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(executable);
        if candidate.is_file() {
            return std::fs::canonicalize(candidate).ok();
        }
    }
    None
}

fn stable_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn query_nvidia_gpus() -> Vec<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok();
    let Some(output) = output.filter(|output| output.status.success()) else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(parse_gpu_line)
        .collect()
}

fn parse_gpu_line(line: &str) -> Option<GpuInfo> {
    let mut parts = line.split(',').map(str::trim);
    Some(GpuInfo {
        index: parts.next()?.parse().ok()?,
        name: parts
            .next()
            .filter(|value| !value.is_empty())
            .map(str::to_string),
        total_vram_mib: parts.next().and_then(|value| value.parse().ok()),
        driver_version: parts
            .next()
            .filter(|value| !value.is_empty())
            .map(str::to_string),
    })
}

fn read_memory_totals() -> Result<(Option<u64>, Option<u64>)> {
    let data = std::fs::read_to_string("/proc/meminfo").context("read /proc/meminfo")?;
    let mut mem_total = None;
    let mut swap_total = None;
    for line in data.lines() {
        if let Some(value) = parse_meminfo_line(line, "MemTotal:") {
            mem_total = Some(value / 1024);
        } else if let Some(value) = parse_meminfo_line(line, "SwapTotal:") {
            swap_total = Some(value / 1024);
        }
    }
    Ok((mem_total, swap_total))
}

fn parse_meminfo_line(line: &str, key: &str) -> Option<u64> {
    line.strip_prefix(key)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_environment_changes() {
        let mut current = EnvironmentSnapshot::default();
        current.llama_server.help_hash = Some("a".to_string());
        let mut same = current.clone();
        assert_eq!(
            compare_environment(Some(&same), &current),
            Compatibility::Current
        );

        same.llama_server.help_hash = Some("b".to_string());
        assert_eq!(
            compare_environment(Some(&same), &current),
            Compatibility::ServerChanged
        );

        same = current.clone();
        same.gpus.push(GpuInfo {
            index: 0,
            name: Some("GPU".to_string()),
            total_vram_mib: Some(1),
            driver_version: Some("1".to_string()),
        });
        assert_eq!(
            compare_environment(Some(&same), &current),
            Compatibility::HardwareChanged
        );
        assert_eq!(
            compare_environment(None, &current),
            Compatibility::LegacyMissingSnapshot
        );
    }
}
