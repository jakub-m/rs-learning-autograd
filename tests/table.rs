use core::fmt;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Write},
};

/// Utility to emit a bunch of vectors as a CSV table.
pub struct Table<F>
where
    F: Copy + fmt::Display,
{
    cols: BTreeMap<String, Vec<F>>,
    max_col_len: usize,
}

impl<F> Table<F>
where
    F: Copy + fmt::Display,
{
    pub fn new() -> Table<F> {
        Table {
            cols: BTreeMap::new(),
            max_col_len: 0,
        }
    }

    /// Accepts [Iterator] so it is possible to use `.map` directly as an argument of [extend_col].
    pub fn extend_col<T>(&mut self, col: &str, values: T)
    where
        T: Iterator<Item = F>,
    {
        self.cols
            .entry(col.to_owned())
            .or_insert(Vec::new())
            .extend(values);
        let col_len = self.cols.get(col).unwrap().len();
        if col_len > self.max_col_len {
            self.max_col_len = col_len
        }
    }

    /// Extend the column with values until the size of the column is equal to the largest column.
    pub fn pad_col(&mut self, col: &str, value: F) {
        let values = self.cols.entry(col.to_owned()).or_insert(Vec::new());
        for _ in 0..(self.max_col_len - values.len()) {
            values.push(value);
        }
        assert_eq!(values.len(), self.max_col_len);
    }

    pub fn to_csv(&self, path: &str) -> io::Result<()> {
        let sep = "\t";
        if self.cols.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "There are no columns to write",
            ));
        }
        self.validate_column_sizes()?;
        let col_names: Vec<String> = self.cols.keys().map(|s| s.to_owned()).collect();
        //let n = self.cols[&col_names[0]].len();

        let mut output = File::create(path)?;
        let row = &col_names.join(sep);
        writeln!(output, "{}", row)?;
        for i in 0..self.max_col_len {
            let row: Vec<String> = col_names
                .iter()
                .map(|c| self.cols.get(c).unwrap().get(i).unwrap())
                .map(|v| format!("{}", v))
                .collect();
            let row = row.join(sep);
            writeln!(output, "{}", row)?;
        }
        Ok(())
    }

    pub fn assert_columns_have_same_lengths(&self) {
        if let Err(e) = self.validate_column_sizes() {
            panic!("{}", e)
        }
    }

    fn validate_column_sizes(&self) -> io::Result<()> {
        let sizes: Vec<usize> = self.cols.keys().map(|key| self.cols[key].len()).collect();
        if sizes.iter().any(|size| size != &self.max_col_len) {
            let s: Vec<String> = self
                .cols
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v.len()))
                .collect();
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Columns have different lengths: {}", s.join(", ")),
            ));
        }
        Ok(())
    }
}
