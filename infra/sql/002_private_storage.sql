-- Plain PostgreSQL can run 001 for offline testing; this migration requires Supabase Storage.
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('agatha-incidents', 'agatha-incidents', false, 52428800, ARRAY['video/mp4','application/json'])
ON CONFLICT (id) DO UPDATE SET public = false, file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;
-- No storage.objects policies: only the server's secret/service role key may upload or sign.
