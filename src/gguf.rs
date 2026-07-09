use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Dense,
    Moe,
    Unknown,
}

/// The on-disk identity used to keep profiler evidence tied to one exact GGUF.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelIdentity {
    pub version: u32,
    pub canonical_path: PathBuf,
    pub file_size_bytes: u64,
    pub modified_at: Option<DateTime<Utc>>,
    pub metadata_hash: String,
}

pub const MODEL_IDENTITY_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum GgufValue {
    String(String),
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufMetadata {
    pub path: PathBuf,
    pub file_name: String,
    pub file_size_bytes: u64,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub size_label: Option<String>,
    pub native_context: Option<u64>,
    pub block_count: Option<u64>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub tokenizer_has_chat_template: bool,
    pub quant: Option<String>,
    pub file_type: Option<u64>,
    pub model_kind: ModelKind,
    pub metadata: BTreeMap<String, GgufValue>,
}

impl GgufMetadata {
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| self.file_name.trim_end_matches(".gguf").to_string())
    }

    pub fn context_or(&self, cap: Option<u64>) -> u64 {
        match (self.native_context, cap) {
            (Some(native), Some(max)) => native.min(max),
            (Some(native), None) => native,
            (None, Some(max)) => max,
            (None, None) => 4096,
        }
    }

    pub fn profiler_dir(&self) -> PathBuf {
        self.model_identity()
            .canonical_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(".llama-cpp-profiler")
            .join("models")
            .join(stable_hash(
                &self.model_identity().canonical_path.to_string_lossy(),
            ))
    }

    /// The pre-beta state location. It is intentionally only used for
    /// diagnostics/legacy discovery, never for new writes or scoring.
    pub fn legacy_profiler_dir(&self) -> PathBuf {
        self.model_identity()
            .canonical_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(".llama-cpp-profiler")
    }

    pub fn model_identity(&self) -> ModelIdentity {
        let canonical_path =
            std::fs::canonicalize(&self.path).unwrap_or_else(|_| self.path.clone());
        let modified_at = std::fs::metadata(&self.path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .map(DateTime::<Utc>::from);
        let metadata_hash = serde_json::to_vec(&self.metadata)
            .map(|bytes| stable_hash_bytes(&bytes))
            .unwrap_or_else(|_| stable_hash(&format!("{:?}", self.metadata)));
        ModelIdentity {
            version: MODEL_IDENTITY_VERSION,
            canonical_path,
            file_size_bytes: self.file_size_bytes,
            modified_at,
            metadata_hash,
        }
    }
}

fn stable_hash(value: &str) -> String {
    stable_hash_bytes(value.as_bytes())
}

fn stable_hash_bytes(value: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanEntry {
    pub path: PathBuf,
    pub file_name: String,
    pub file_size_bytes: u64,
    pub architecture: Option<String>,
    pub quant: Option<String>,
    pub model_kind: ModelKind,
    pub native_context: Option<u64>,
    pub latest_recommendation: Option<String>,
}

pub fn discover_models(root: impl AsRef<Path>) -> Result<Vec<GgufMetadata>> {
    let root = root.as_ref();
    let mut files = Vec::new();
    collect_gguf_files(root, &mut files)?;
    files.sort();

    let mut models = Vec::new();
    for file in files {
        match read_metadata(&file) {
            Ok(metadata) => models.push(metadata),
            Err(err) => {
                eprintln!("skipping {}: {err:#}", file.display());
            }
        }
    }
    Ok(models)
}

pub fn scan_entries(root: impl AsRef<Path>) -> Result<Vec<ScanEntry>> {
    let mut entries = Vec::new();
    for model in discover_models(root)? {
        let latest_recommendation = latest_recommendation_label(&model.profiler_dir());
        entries.push(ScanEntry {
            path: model.path,
            file_name: model.file_name,
            file_size_bytes: model.file_size_bytes,
            architecture: model.architecture,
            quant: model.quant,
            model_kind: model.model_kind,
            native_context: model.native_context,
            latest_recommendation,
        });
    }
    Ok(entries)
}

pub fn resolve_model_path(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    if path.is_file() {
        if is_model_gguf(path) {
            return Ok(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()));
        }
        bail!("not a model GGUF: {}", path.display());
    }

    if !path.is_dir() {
        bail!("path does not exist: {}", path.display());
    }

    let mut files = Vec::new();
    collect_gguf_files(path, &mut files)?;
    if files.is_empty() {
        bail!("no model GGUF files found under {}", path.display());
    }

    files.sort_by_key(|path| {
        std::fs::metadata(path)
            .map(|metadata| std::cmp::Reverse(metadata.len()))
            .unwrap_or(std::cmp::Reverse(0))
    });
    let selected = files.remove(0);
    Ok(std::fs::canonicalize(&selected).unwrap_or(selected))
}

pub fn read_metadata(path: impl AsRef<Path>) -> Result<GgufMetadata> {
    let path = path.as_ref();
    let canonical_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let path = canonical_path.as_path();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_size_bytes = file
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let mut reader = BufReader::new(file);

    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        bail!("invalid GGUF magic");
    }

    let gguf_version = read_u32(&mut reader)?;
    let tensor_count = read_u64(&mut reader)?;
    let metadata_kv_count = read_u64(&mut reader)?;
    let mut metadata = BTreeMap::new();

    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        if let Some(value) = read_value_or_skip(&mut reader, value_type)? {
            metadata.insert(key, value);
        }
    }

    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_string();
    let architecture = string_value(&metadata, "general.architecture");
    let name = string_value(&metadata, "general.name");
    let size_label = string_value(&metadata, "general.size_label");
    let native_context = first_u64_with_suffix(&metadata, ".context_length");
    let block_count = first_u64_with_suffix(&metadata, ".block_count");
    let expert_count = first_u64_with_suffix(&metadata, ".expert_count");
    let expert_used_count = first_u64_with_suffix(&metadata, ".expert_used_count");
    let tokenizer_has_chat_template = metadata.contains_key("tokenizer.chat_template")
        || metadata
            .keys()
            .any(|key| key.ends_with(".chat_template") || key.contains("chat_template"));
    let file_type = u64_value(&metadata, "general.file_type");
    let quant =
        infer_quant_from_filename(&file_name).or_else(|| file_type.and_then(quant_from_file_type));
    let model_kind = if expert_count.unwrap_or(0) > 0
        || file_name.to_ascii_lowercase().contains("moe")
        || architecture
            .as_deref()
            .map(|arch| arch.to_ascii_lowercase().contains("moe"))
            .unwrap_or(false)
    {
        ModelKind::Moe
    } else if architecture.is_some() || block_count.is_some() {
        ModelKind::Dense
    } else {
        ModelKind::Unknown
    };

    Ok(GgufMetadata {
        path: path.to_path_buf(),
        file_name,
        file_size_bytes,
        gguf_version,
        tensor_count,
        metadata_kv_count,
        name,
        architecture,
        size_label,
        native_context,
        block_count,
        expert_count,
        expert_used_count,
        tokenizer_has_chat_template,
        quant,
        file_type,
        model_kind,
        metadata,
    })
}

pub fn infer_quant_from_filename(file_name: &str) -> Option<String> {
    let re = Regex::new(r"(?i)(IQ[0-9]+(?:_[A-Z0-9]+)+|Q[0-9]+(?:_[A-Z0-9]+)+|BF16|F16|F32)")
        .expect("valid quant regex");
    re.captures(file_name)
        .and_then(|captures| captures.get(1))
        .map(|matched| matched.as_str().to_ascii_uppercase())
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn collect_gguf_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if root.is_file() {
        if is_model_gguf(root) {
            files.push(root.to_path_buf());
        }
        return Ok(());
    }

    for entry in std::fs::read_dir(root).with_context(|| format!("read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if path
                .file_name()
                .and_then(OsStr::to_str)
                .map(|name| name == ".llama-cpp-profiler" || name == ".git")
                .unwrap_or(false)
            {
                continue;
            }
            collect_gguf_files(&path, files)?;
        } else if is_model_gguf(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_model_gguf(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    if !file_name.to_ascii_lowercase().ends_with(".gguf") {
        return false;
    }
    let lower = file_name.to_ascii_lowercase();
    !lower.contains("mmproj") && !lower.contains("draft")
}

fn latest_recommendation_label(profiler_dir: &Path) -> Option<String> {
    let path = profiler_dir.join("recommendations.json");
    let data = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&data).ok()?;
    value
        .get("profiles")
        .and_then(|profiles| profiles.as_array())
        .and_then(|profiles| profiles.first())
        .and_then(|profile| profile.get("id"))
        .and_then(|id| id.as_str())
        .map(str::to_string)
}

fn read_value_or_skip<R: Read + Seek>(
    reader: &mut R,
    value_type: u32,
) -> Result<Option<GgufValue>> {
    match value_type {
        0 => Ok(Some(GgufValue::U64(read_u8(reader)? as u64))),
        1 => Ok(Some(GgufValue::I64(read_i8(reader)? as i64))),
        2 => Ok(Some(GgufValue::U64(read_u16(reader)? as u64))),
        3 => Ok(Some(GgufValue::I64(read_i16(reader)? as i64))),
        4 => Ok(Some(GgufValue::U64(read_u32(reader)? as u64))),
        5 => Ok(Some(GgufValue::I64(read_i32(reader)? as i64))),
        6 => Ok(Some(GgufValue::F64(read_f32(reader)? as f64))),
        7 => Ok(Some(GgufValue::Bool(read_u8(reader)? != 0))),
        8 => Ok(Some(GgufValue::String(read_gguf_string(reader)?))),
        9 => {
            skip_array(reader)?;
            Ok(None)
        }
        10 => Ok(Some(GgufValue::U64(read_u64(reader)?))),
        11 => Ok(Some(GgufValue::I64(read_i64(reader)?))),
        12 => Ok(Some(GgufValue::F64(read_f64(reader)?))),
        other => Err(anyhow!("unsupported GGUF value type {other}")),
    }
}

fn skip_array<R: Read + Seek>(reader: &mut R) -> Result<()> {
    let element_type = read_u32(reader)?;
    let length = read_u64(reader)?;
    match element_type {
        0 | 1 | 7 => skip_bytes(reader, length)?,
        2 | 3 => skip_bytes(
            reader,
            length.checked_mul(2).context("array size overflow")?,
        )?,
        4..=6 => skip_bytes(
            reader,
            length.checked_mul(4).context("array size overflow")?,
        )?,
        10..=12 => skip_bytes(
            reader,
            length.checked_mul(8).context("array size overflow")?,
        )?,
        8 => {
            for _ in 0..length {
                let len = read_u64(reader)?;
                skip_bytes(reader, len)?;
            }
        }
        other => bail!("unsupported GGUF array element type {other}"),
    }
    Ok(())
}

fn skip_bytes<R: Seek>(reader: &mut R, bytes: u64) -> Result<()> {
    let offset = i64::try_from(bytes).context("skip offset too large")?;
    reader.seek(SeekFrom::Current(offset))?;
    Ok(())
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64(reader)?;
    let len = usize::try_from(len).context("GGUF string too large")?;
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("GGUF string is not UTF-8")
}

fn read_exact<const N: usize, R: Read>(reader: &mut R) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    Ok(read_exact::<1, _>(reader)?[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8> {
    Ok(read_exact::<1, _>(reader)?[0] as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16> {
    Ok(u16::from_le_bytes(read_exact(reader)?))
}

fn read_i16<R: Read>(reader: &mut R) -> Result<i16> {
    Ok(i16::from_le_bytes(read_exact(reader)?))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact(reader)?))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    Ok(i32::from_le_bytes(read_exact(reader)?))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact(reader)?))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    Ok(i64::from_le_bytes(read_exact(reader)?))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    Ok(f32::from_le_bytes(read_exact(reader)?))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    Ok(f64::from_le_bytes(read_exact(reader)?))
}

fn string_value(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<String> {
    match metadata.get(key) {
        Some(GgufValue::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn u64_value(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<u64> {
    match metadata.get(key) {
        Some(GgufValue::U64(value)) => Some(*value),
        Some(GgufValue::I64(value)) if *value >= 0 => Some(*value as u64),
        _ => None,
    }
}

fn first_u64_with_suffix(metadata: &BTreeMap<String, GgufValue>, suffix: &str) -> Option<u64> {
    metadata
        .iter()
        .find(|(key, _)| key.ends_with(suffix))
        .and_then(|(key, _)| u64_value(metadata, key))
}

fn quant_from_file_type(file_type: u64) -> Option<String> {
    let quant = match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        33 => "TQ1_0",
        34 => "TQ2_0",
        35 => "Q4_0_4_4",
        36 => "Q4_0_4_8",
        37 => "Q4_0_8_8",
        38 => "MXFP4",
        _ => return None,
    };
    Some(quant.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn infers_quant_from_common_file_names() {
        assert_eq!(
            infer_quant_from_filename("Qwopus-35B-A3B-Q4_K_P.gguf"),
            Some("Q4_K_P".to_string())
        );
        assert_eq!(
            infer_quant_from_filename("model-IQ4_XS.gguf"),
            Some("IQ4_XS".to_string())
        );
        assert_eq!(
            infer_quant_from_filename("thing.Q5_K_M.gguf"),
            Some("Q5_K_M".to_string())
        );
        assert_eq!(infer_quant_from_filename("qwen3-model.gguf"), None);
    }

    #[test]
    fn parses_minimal_gguf_metadata() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&6u64.to_le_bytes()).unwrap();
        write_kv_string(&mut file, "general.architecture", "qwen2");
        write_kv_string(&mut file, "general.name", "Tiny Test");
        write_kv_u32(&mut file, "general.file_type", 15);
        write_kv_u32(&mut file, "qwen2.context_length", 262_144);
        write_kv_u32(&mut file, "qwen2.expert_count", 16);
        write_kv_string(&mut file, "tokenizer.chat_template", "{{ messages }}");
        file.flush().unwrap();

        let metadata = read_metadata(file.path()).unwrap();
        assert_eq!(metadata.gguf_version, 3);
        assert_eq!(metadata.architecture.as_deref(), Some("qwen2"));
        assert_eq!(metadata.name.as_deref(), Some("Tiny Test"));
        assert_eq!(metadata.native_context, Some(262_144));
        assert_eq!(metadata.expert_count, Some(16));
        assert_eq!(metadata.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(metadata.model_kind, ModelKind::Moe);
        assert!(metadata.tokenizer_has_chat_template);
    }

    #[test]
    fn profiler_state_isolated_for_models_in_one_directory() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first.gguf");
        let second = directory.path().join("second.gguf");
        write_test_gguf(&first, "first", 4096);
        write_test_gguf(&second, "second", 8192);

        let first_metadata = read_metadata(&first).unwrap();
        let second_metadata = read_metadata(&second).unwrap();
        assert_ne!(
            first_metadata.profiler_dir(),
            second_metadata.profiler_dir()
        );
        assert!(
            first_metadata
                .profiler_dir()
                .parent()
                .unwrap()
                .ends_with("models")
        );
        assert_eq!(
            first_metadata.model_identity().canonical_path,
            std::fs::canonicalize(first).unwrap()
        );
    }

    fn write_test_gguf(path: &Path, name: &str, context: u32) {
        let mut file = File::create(path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&3u64.to_le_bytes()).unwrap();
        write_identity_key(&mut file, "general.architecture", 8);
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name.as_bytes()).unwrap();
        write_identity_key(&mut file, "general.name", 8);
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name.as_bytes()).unwrap();
        write_identity_key(&mut file, "llama.context_length", 4);
        file.write_all(&context.to_le_bytes()).unwrap();
    }

    fn write_identity_key(file: &mut File, key: &str, kind: u32) {
        file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        file.write_all(key.as_bytes()).unwrap();
        file.write_all(&kind.to_le_bytes()).unwrap();
    }

    fn write_key(file: &mut tempfile::NamedTempFile, key: &str, kind: u32) {
        file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        file.write_all(key.as_bytes()).unwrap();
        file.write_all(&kind.to_le_bytes()).unwrap();
    }

    fn write_kv_string(file: &mut tempfile::NamedTempFile, key: &str, value: &str) {
        write_key(file, key, 8);
        file.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }

    fn write_kv_u32(file: &mut tempfile::NamedTempFile, key: &str, value: u32) {
        write_key(file, key, 4);
        file.write_all(&value.to_le_bytes()).unwrap();
    }
}
