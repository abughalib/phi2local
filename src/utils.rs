use std::env;

pub enum MyResult<T> {
    Ok(T),
}

pub fn get_pg_url() -> String {
    match env::var("PHI2DB") {
        Ok(res) => {
            return res;
        }
        Err(_e) => {
            panic!("Set PHI2DB environment variable");
        }
    }
}
