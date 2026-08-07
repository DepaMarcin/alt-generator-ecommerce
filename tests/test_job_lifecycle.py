"""Regression coverage for the "zadanie nie istnieje lub wygasło" 404 bug
(a still-running job's JOBS entry was being reaped purely by age, regardless
of status) and the per-batch image dedup cache added alongside it.
"""
import os
import shutil
import time
import uuid

import pytest

import app


@pytest.fixture
def registered_job():
    """Registers a minimal job in the real app.JOBS dict (like the
    /generate-alt route does) and guarantees it's removed again after the
    test, so these tests never leak state into others."""
    task_id = str(uuid.uuid4())

    def _register(status: str, created_at: float = None):
        with app.JOBS_LOCK:
            app.JOBS[task_id] = {
                "status": status,
                "total": 0,
                "processed": 0,
                "success_count": 0,
                "error_count": 0,
                "results": [],
                "error_message": None,
                "created_at": created_at if created_at is not None else time.time(),
                "image_cache": {},
            }
        return task_id

    yield _register

    with app.JOBS_LOCK:
        app.JOBS.pop(task_id, None)


class TestCleanOldJobsNeverDropsActiveJobs:
    @pytest.mark.parametrize("active_status", ["parsing", "processing"])
    def test_active_job_survives_even_when_very_old(self, registered_job, active_status):
        task_id = registered_job(active_status, created_at=time.time() - (app.JOB_MAX_AGE_SECONDS * 10))
        app.clean_old_jobs()
        assert task_id in app.JOBS

    @pytest.mark.parametrize("finished_status", ["completed", "error", "stopped_error"])
    def test_finished_job_is_reaped_once_past_max_age(self, registered_job, finished_status):
        task_id = registered_job(finished_status, created_at=time.time() - (app.JOB_MAX_AGE_SECONDS + 60))
        app.clean_old_jobs()
        assert task_id not in app.JOBS

    @pytest.mark.parametrize("finished_status", ["completed", "error", "stopped_error"])
    def test_finished_job_survives_until_max_age_is_reached(self, registered_job, finished_status):
        task_id = registered_job(finished_status, created_at=time.time())
        app.clean_old_jobs()
        assert task_id in app.JOBS


class TestGetStatusEndpoint:
    def test_missing_job_returns_404_with_expected_message(self):
        client = app.app.test_client()
        response = client.get(f"/status/{uuid.uuid4()}")
        assert response.status_code == 404
        assert "nie istnieje" in response.get_json()["error"]

    def test_response_never_leaks_the_internal_image_cache(self, registered_job):
        task_id = registered_job("processing")
        with app.JOBS_LOCK:
            app.JOBS[task_id]["image_cache"]["https://cdn.sklep.pl/img/logo.png"] = {"alt": "cached"}

        client = app.app.test_client()
        response = client.get(f"/status/{task_id}")

        assert response.status_code == 200
        assert "image_cache" not in response.get_json()


class TestImageCacheDeduplication:
    def test_second_call_for_same_url_skips_download_and_openai(
        self, registered_job, mocker, openai_stub, dummy_image_path
    ):
        task_id = registered_job("processing")
        image_url = "https://cdn.sklep.pl/img/shared-logo.png"

        mock_download = mocker.patch.object(app, "download_image_from_url", return_value=dummy_image_path)
        mock_create = openai_stub("Logo sklepu na białym tle.")

        first = app.process_single_image(image_url, "Produkt: A", "/tmp/job", task_id=task_id)
        second = app.process_single_image(image_url, "Produkt: B", "/tmp/job", task_id=task_id)

        assert mock_download.call_count == 1
        assert mock_create.call_count == 1
        assert second["alt"] == first["alt"]
        assert second["image_data"] == first["image_data"]

    def test_different_urls_are_not_conflated(
        self, registered_job, mocker, openai_stub, dummy_image_path, tmp_path
    ):
        task_id = registered_job("processing")

        # Two independent (uncached) downloads each need their own real file
        # on disk - compress_image() deletes/renames its input in place, so
        # reusing a single path across two calls would break the second one.
        def _fresh_copy(_url, dest_dir):
            dest = os.path.join(dest_dir, f"{uuid.uuid4().hex}.jpg")
            shutil.copy(dummy_image_path, dest)
            return dest

        mocker.patch.object(app, "download_image_from_url", side_effect=_fresh_copy)
        mock_create = openai_stub(["Logo sklepu.", "Ikona dostawy."])

        job_dir = str(tmp_path)
        first = app.process_single_image(
            "https://cdn.sklep.pl/img/logo.png", "Produkt: A", job_dir, task_id=task_id
        )
        second = app.process_single_image(
            "https://cdn.sklep.pl/img/delivery-icon.png", "Produkt: A", job_dir, task_id=task_id
        )

        assert mock_create.call_count == 2
        assert first["alt"] != second["alt"]

    def test_failed_downloads_are_never_cached(
        self, registered_job, mocker, openai_stub, dummy_image_path
    ):
        task_id = registered_job("processing")
        image_url = "https://cdn.sklep.pl/img/flaky.png"
        mock_download = mocker.patch.object(
            app, "download_image_from_url",
            side_effect=[ValueError("404"), dummy_image_path],
        )
        openai_stub("Logo sklepu.")

        with pytest.raises(ValueError):
            app.process_single_image(image_url, "Produkt: A", "/tmp/job", task_id=task_id)

        result = app.process_single_image(image_url, "Produkt: A", "/tmp/job", task_id=task_id)

        assert mock_download.call_count == 2
        assert result["skipped"] is False
        assert result["alt"] == "Logo sklepu."
