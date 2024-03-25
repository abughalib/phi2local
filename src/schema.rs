// @generated automatically by Diesel CLI.

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::*;

    items (id) {
        id -> Int4,
        chunk_number -> Nullable<Int4>,
        title -> Text,
        content -> Nullable<Text>,
        embedding -> Nullable<Vector>,
    }
}
