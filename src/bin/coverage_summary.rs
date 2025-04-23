use serde::Deserialize;
use std::{
    error::Error,
    fs::File,
    io::{BufReader, Write},
};
use tabwriter::TabWriter;

#[derive(Deserialize)]
struct Root {
    files: Option<Vec<FileCov>>,
    data:   Option<Vec<DataEntry>>,
}

#[derive(Deserialize)]
struct DataEntry {
    files: Vec<FileCov>,
}

#[derive(Deserialize)]
struct FileCov {
    filename: String,
    summary:  Summary,
}

#[derive(Deserialize)]
struct Summary {
    functions: Pct,
    lines:     Pct,
    regions:   Pct,
}

#[derive(Deserialize)]
struct Pct {
    percent: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("cov.json")?;
    let root: Root = serde_json::from_reader(BufReader::new(file))?;

    // Decide where the files array is
    let files = if let Some(files) = root.files {
        files
    } else if let Some(data_entries) = root.data {
        data_entries.into_iter()
            .flat_map(|de| de.files.into_iter())
            .collect()
    } else {
        eprintln!("Error: no `files` or `data` in cov.json");
        std::process::exit(1);
    };

    let mut tw = TabWriter::new(std::io::stdout())
        .padding(2)
        .minwidth(10);

    writeln!(tw, "File\tFuncs %\tLines %\tRegions %")?;
    for f in files {
        let name = f.filename.rsplit('/').next().unwrap_or(&f.filename);
        writeln!(
            tw,
            "{}\t{:.1}%\t{:.1}%\t{:.1}%",
            name,
            f.summary.functions.percent,
            f.summary.lines.percent,
            f.summary.regions.percent
        )?;
    }
    tw.flush()?;
    Ok(())
}