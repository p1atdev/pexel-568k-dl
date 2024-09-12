use anyhow::{bail, Context, Result};
use clap::Parser;
use futures::future;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{Method, Url};
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

const MAX_LENGTH: u32 = 2048;

#[derive(Parser, Debug, Clone)]
struct Cli {
    /// data.jsonl path
    #[arg(default_value = "data.jsonl")]
    data: String,

    #[arg(short, long, default_value = "output")]
    output: String,
}

#[derive(Deserialize, Clone)]
struct Row {
    id: u32,
    url: String,
    // cogvlm: String,
    internvl2: String,
    width: u32,
    height: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    let data_path = args.data;
    let output_path = args.output;

    let data = tokio::fs::read(data_path).await?;
    let data = String::from_utf8(data)?;
    let data: Vec<Row> = data
        .lines()
        .map(|line| {
            let row: Row = serde_json::from_str(line).unwrap();
            row
        })
        .collect();

    let client = Arc::new(reqwest::Client::new());

    let output_path = Path::new(&output_path).to_path_buf();
    tokio::fs::create_dir_all(&output_path).await?;

    let output_path = Arc::new(Path::new(&output_path).to_path_buf());

    let bar = ProgressBar::new(data.len() as u64);
    bar.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:50.cyan/blue} {pos:>7}/{len:7} {msg}",
    )?);
    let shared_bar = Arc::new(tokio::sync::Mutex::new(bar));

    futures::stream::iter(data)
        .filter(|row| {
            let id = row.id.clone();
            let image_path = output_path.join(id.to_string());

            let exists = image_path.with_extension("jpg").exists();

            future::ready(!exists)
        })
        .map(|row| {
            let cloned_client = client.clone();

            tokio::spawn(async move {
                let url = Url::parse(&row.url).context("Failed to parse URL")?;
                let response = cloned_client
                    .request(Method::GET, url)
                    .send()
                    .await
                    .context("Failed to send request")?;
                let bytes = response
                    .bytes()
                    .await
                    .context("Failed to get response body")?;

                Result::<_>::Ok((bytes, row))
            })
        })
        .buffer_unordered(16)
        .map(|pair| pair?)
        .map(|pair| {
            tokio::task::spawn_blocking(move || {
                let (bytes, row) = pair?;

                let mut image = photon_rs::native::open_image_from_bytes(bytes.deref())
                    .context("Failed to open image")?;

                let width = row.width;
                let height = row.height;

                if width > MAX_LENGTH || height > MAX_LENGTH {
                    // resize
                    let ratio = width as f32 / height as f32;
                    let (new_width, new_height) = if width > height {
                        (MAX_LENGTH, (MAX_LENGTH as f32 / ratio) as u32)
                    } else {
                        ((MAX_LENGTH as f32 * ratio) as u32, MAX_LENGTH)
                    };

                    image = photon_rs::transform::resize(
                        &image,
                        new_width,
                        new_height,
                        photon_rs::transform::SamplingFilter::Nearest,
                    );
                }

                Result::<_>::Ok((image, row))
            })
        })
        .buffer_unordered(32)
        .map(|pair| pair?)
        .map(|pair| {
            let cloned_output_path = output_path.clone();
            let bar = shared_bar.clone();

            tokio::spawn(async move {
                let (image, row) = pair?;
                let id = row.id.clone();

                let image_path = cloned_output_path.join(id.to_string());
                let image_path = image_path.with_extension("jpg");

                let mut file = tokio::fs::File::create(image_path)
                    .await
                    .context("Failed to create file")?;
                let bytes = image.get_bytes_jpeg(80);

                file.write_all(&bytes).await?;

                let caption = row.internvl2.clone();

                let mut caption_path = cloned_output_path.join(id.to_string());
                caption_path.set_extension("txt");

                let mut file = tokio::fs::File::create(caption_path)
                    .await
                    .context("Failed to create file")?;
                file.write_all(caption.as_bytes()).await?;

                bar.lock().await.inc(1);

                Result::<_>::Ok(())
            })
        })
        .buffer_unordered(32)
        .map(|task| task)
        .collect::<Vec<_>>()
        .await;

    shared_bar.lock().await.finish();

    Ok(())
}
