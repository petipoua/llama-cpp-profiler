use crate::gguf::{ScanEntry, format_bytes};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
};
use std::io;
use std::path::PathBuf;
use std::time::Duration;

pub fn pick_scan_entry(entries: &[ScanEntry]) -> Result<Option<PathBuf>> {
    if entries.is_empty() {
        println!("No GGUF models found.");
        return Ok(None);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_picker(&mut terminal, entries);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn run_picker(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    entries: &[ScanEntry],
) -> Result<Option<PathBuf>> {
    let mut state = PickerState::new(entries.len());

    loop {
        let filtered = filtered_indices(entries, &state.search);
        if state.selected >= filtered.len() {
            state.selected = filtered.len().saturating_sub(1);
        }

        terminal.draw(|frame| {
            let area = frame.area();
            frame.render_widget(Clear, area);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(8),
                    Constraint::Length(4),
                ])
                .split(area);

            let search = Paragraph::new(Line::from(vec![
                Span::styled("Search ", Style::default().fg(Color::Cyan)),
                Span::raw(&state.search),
                Span::styled(
                    "  / clears with Backspace",
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("llama-cpp-profiler scan"),
            );
            frame.render_widget(search, chunks[0]);

            let items = filtered
                .iter()
                .map(|index| {
                    let entry = &entries[*index];
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            entry.file_name.clone(),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::raw("  "),
                        Span::styled(
                            format_bytes(entry.file_size_bytes),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::raw("  "),
                        Span::styled(
                            entry.quant.clone().unwrap_or_else(|| "-".to_string()),
                            Style::default().fg(Color::Yellow),
                        ),
                        Span::raw("  "),
                        Span::styled(
                            format!("{:?}", entry.model_kind).to_ascii_lowercase(),
                            Style::default().fg(Color::Magenta),
                        ),
                        Span::raw("  "),
                        Span::styled(
                            entry
                                .latest_recommendation
                                .clone()
                                .unwrap_or_else(|| "no recommendation".to_string()),
                            Style::default().fg(Color::Green),
                        ),
                    ]))
                })
                .collect::<Vec<_>>();

            let mut list_state = ListState::default();
            if !filtered.is_empty() {
                list_state.select(Some(state.selected));
            }
            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title("Models"))
                .highlight_symbol(">> ")
                .highlight_style(
                    Style::default()
                        .bg(Color::DarkGray)
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                );
            frame.render_stateful_widget(list, chunks[1], &mut list_state);

            let detail = filtered
                .get(state.selected)
                .map(|index| {
                    let entry = &entries[*index];
                    format!(
                        "{}\narch: {}  ctx: {}  path: {}",
                        entry
                            .latest_recommendation
                            .as_deref()
                            .unwrap_or("no saved recommendation"),
                        entry.architecture.as_deref().unwrap_or("-"),
                        entry
                            .native_context
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        entry.path.display()
                    )
                })
                .unwrap_or_else(|| "No matches".to_string());
            let footer = Paragraph::new(format!("{detail}\nEnter select  arrows move  q quit"))
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::ALL).title("Status"));
            frame.render_widget(footer, chunks[2]);
        })?;

        if !event::poll(Duration::from_millis(200))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
            KeyCode::Enter => {
                let filtered = filtered_indices(entries, &state.search);
                return Ok(filtered
                    .get(state.selected)
                    .map(|index| entries[*index].path.clone()));
            }
            KeyCode::Down => {
                state.selected = (state.selected + 1).min(filtered.len().saturating_sub(1))
            }
            KeyCode::Up => state.selected = state.selected.saturating_sub(1),
            KeyCode::Backspace => {
                state.search.pop();
                state.selected = 0;
            }
            KeyCode::Char(ch) => {
                state.search.push(ch);
                state.selected = 0;
            }
            _ => {}
        }
    }
}

struct PickerState {
    search: String,
    selected: usize,
}

impl PickerState {
    fn new(_entry_count: usize) -> Self {
        Self {
            search: String::new(),
            selected: 0,
        }
    }
}

fn filtered_indices(entries: &[ScanEntry], search: &str) -> Vec<usize> {
    if search.trim().is_empty() {
        return (0..entries.len()).collect();
    }
    let search = search.to_ascii_lowercase();
    entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let haystack = format!(
                "{} {} {} {}",
                entry.file_name,
                entry.path.display(),
                entry.quant.as_deref().unwrap_or_default(),
                entry.architecture.as_deref().unwrap_or_default()
            )
            .to_ascii_lowercase();
            haystack.contains(&search).then_some(index)
        })
        .collect()
}
