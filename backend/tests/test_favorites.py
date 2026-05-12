import os
import tempfile
import unittest
from typing import cast

from flask import Flask
from sqlalchemy import text


def _set_default_env() -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["MARIADB_USER"] = "root"
    os.environ["MARIADB_PASSWORD"] = "password"
    os.environ["MARIADB_HOST"] = "127.0.0.1"
    os.environ["MARIADB_DATABASE"] = "pdx"
    os.environ["AUTH_SECRET_KEY"] = "test-auth-secret"
    os.environ["PUBLIC_API_BASE_URL"] = "http://localhost:5000"
    os.environ["PUBLIC_FRONTEND_BASE_URL"] = "http://localhost:8080"
    os.environ["TURNSTILE_ENABLED"] = "0"


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_fav_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ["AUTH_DATABASE_URI"] = f"sqlite:///{_AUTH_DB_TEMP.name}"


from backend.auth.runtime import AccessContext  # noqa: E402
from backend.app import create_test_app  # noqa: E402
from backend.extensions import db  # noqa: E402
from backend.models import Agreements, Clauses, Favorite, FavoriteProject, FavoriteProjectAssignment, FavoriteTag, FavoriteTagAssignment, Sections  # noqa: E402
from backend.models.auth import AuthUser  # noqa: E402
from backend.routes.favorites import FavoritesDeps, register_favorites_routes  # noqa: E402
import backend.app as backend_app  # noqa: E402


class FavoritesRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            }
        )
        with cls.app.app_context():
            db.create_all(bind_key="auth")

    def setUp(self) -> None:
        _set_default_env()
        with self.app.app_context():
            engine = db.engines["auth"]
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM favorite_tag_assignments"))
                conn.execute(text("DELETE FROM favorite_project_assignments"))
                conn.execute(text("DELETE FROM favorite_tags"))
                conn.execute(text("DELETE FROM favorites"))
                conn.execute(text("DELETE FROM favorite_projects"))
                conn.execute(text("DELETE FROM auth_sessions"))
                conn.execute(text("DELETE FROM auth_users"))
        backend_app._rate_limit_state.clear()
        backend_app._endpoint_rate_limit_state.clear()

    def _create_user_and_bearer(self, email: str = "u@example.com") -> str:
        with self.app.app_context():
            user = AuthUser()
            user.email = email
            user.password_hash = backend_app.generate_password_hash("password123")
            user.email_verified_at = backend_app._utc_now()
            db.session.add(user)
            db.session.commit()
            with self.app.test_request_context("/v1/me/favorites"):
                return backend_app._issue_session_token(user.id)

    def _client_with_bearer(self):
        token = self._create_user_and_bearer()
        client = self.app.test_client()
        client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token}"
        return client

    def _client_with_verified_non_user_tier(self, tier: str = "api_key"):
        user = AuthUser()
        user.id = "verified-non-user"
        user.email = "tiered@example.com"
        user.email_verified_at = backend_app._utc_now()

        test_app = Flask(__name__)
        test_app.config.update(
            TESTING=True,
            SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
            SQLALCHEMY_BINDS={"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        db.init_app(test_app)
        register_favorites_routes(
            test_app,
            deps=FavoritesDeps(
                Favorite=Favorite,
                FavoriteProject=FavoriteProject,
                FavoriteProjectAssignment=FavoriteProjectAssignment,
                FavoriteTag=FavoriteTag,
                FavoriteTagAssignment=FavoriteTagAssignment,
                Sections=Sections,
                Clauses=Clauses,
                Agreements=Agreements,
                db=db,
                _require_auth_db=lambda: None,
                _require_verified_user=lambda: (
                    user,
                    AccessContext(tier=tier, user_id=user.id),
                ),
                _auth_is_mocked=lambda: False,
                _load_json=backend_app._load_json,
                _utc_now=backend_app._utc_now,
            ),
        )
        return test_app.test_client()

    def test_unauthenticated_listing_returns_401(self):
        client = self.app.test_client()
        res = client.get("/v1/me/favorites")
        self.assertEqual(res.status_code, 401)

    def test_api_key_access_is_forbidden(self):
        with self.app.app_context():
            user = AuthUser()
            user.email = "api-key@example.com"
            user.password_hash = backend_app.generate_password_hash("password123")
            user.email_verified_at = backend_app._utc_now()
            db.session.add(user)
            db.session.commit()
            _, api_key = backend_app._create_api_key(user_id=user.id, name="favorites")
        client = self.app.test_client()
        res = client.get("/v1/me/favorites", headers={"X-API-Key": api_key})
        self.assertIn(res.status_code, {401, 403})

    def test_verified_api_key_tier_is_rejected_for_listing(self):
        client = self._client_with_verified_non_user_tier()
        res = client.get("/v1/me/favorites")
        self.assertEqual(res.status_code, 403)
        self.assertIn("signed-in user session", res.get_data(as_text=True))

    def test_verified_non_user_tier_is_rejected_for_create(self):
        client = self._client_with_verified_non_user_tier(tier="mcp")
        res = client.post(
            "/v1/me/favorites",
            json={
                "item_type": "agreement",
                "item_uuid": "66666666-6666-6666-6666-666666666666",
            },
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("signed-in user session", res.get_data(as_text=True))

    def test_listing_creates_default_project_lazily(self):
        client = self._client_with_bearer()
        res = client.get("/v1/me/favorite-projects")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        projects = cast(list[dict[str, object]], body["projects"])
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["name"], "Scratchpad")
        self.assertTrue(projects[0]["is_default"])

    def test_create_then_idempotent_then_list(self):
        client = self._client_with_bearer()
        item_uuid = "11111111-1111-1111-1111-111111111111"

        res = client.post(
            "/v1/me/favorites",
            json={
                "item_type": "section",
                "item_uuid": item_uuid,
                "context": {"page": "/search?q=x"},
            },
        )
        self.assertEqual(res.status_code, 201)
        body = res.get_json()
        self.assertTrue(body["created"])
        self.assertEqual(body["favorite"]["item_type"], "section")
        first_id = body["favorite"]["id"]

        res = client.post(
            "/v1/me/favorites",
            json={
                "item_type": "section",
                "item_uuid": item_uuid,
                "note": "second pass",
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertFalse(body["created"])
        self.assertEqual(body["favorite"]["id"], first_id)
        self.assertEqual(body["favorite"]["note"], "second pass")

        res = client.get("/v1/me/favorites")
        self.assertEqual(res.status_code, 200)
        favs = cast(list[dict[str, object]], res.get_json()["favorites"])
        self.assertEqual(len(favs), 1)
        self.assertEqual(favs[0]["id"], first_id)

    def test_create_minimal_response_skips_rich_fields(self):
        client = self._client_with_bearer()
        item_uuid = "22222222-2222-2222-2222-222222222222"
        res = client.post(
            "/v1/me/favorites?view=minimal",
            json={"item_type": "agreement", "item_uuid": item_uuid},
        )
        self.assertEqual(res.status_code, 201)
        body = res.get_json()
        self.assertTrue(body["created"])
        favorite = cast(dict[str, object], body["favorite"])
        self.assertEqual(favorite["item_type"], "agreement")
        self.assertEqual(favorite["item_uuid"], item_uuid)
        self.assertNotIn("agreement_uuid", favorite)
        self.assertNotIn("tags", favorite)

    def test_exists_lookup(self):
        client = self._client_with_bearer()
        starred = "22222222-2222-2222-2222-222222222222"
        not_starred = "33333333-3333-3333-3333-333333333333"
        client.post(
            "/v1/me/favorites",
            json={"item_type": "agreement", "item_uuid": starred},
        )
        res = client.get(
            f"/v1/me/favorites/exists?item_type=agreement&item_uuids={starred},{not_starred}"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIn(starred, body["favorites"])
        self.assertNotIn(not_starred, body["favorites"])

    def test_patch_note_and_delete(self):
        client = self._client_with_bearer()
        res = client.post(
            "/v1/me/favorites",
            json={
                "item_type": "tax_clause",
                "item_uuid": "44444444-4444-4444-4444-444444444444",
            },
        )
        favorite_id = res.get_json()["favorite"]["id"]

        res = client.patch(
            f"/v1/me/favorites/{favorite_id}", json={"note": "  hello  "}
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["favorite"]["note"], "hello")

        res = client.delete(f"/v1/me/favorites/{favorite_id}")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["deleted"])

        res = client.get("/v1/me/favorites")
        self.assertEqual(res.get_json()["favorites"], [])

    def test_user_isolation(self):
        client_a = self._client_with_bearer()
        token_b = self._create_user_and_bearer(email="b@example.com")
        client_b = self.app.test_client()
        client_b.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token_b}"

        res = client_a.post(
            "/v1/me/favorites",
            json={
                "item_type": "agreement",
                "item_uuid": "55555555-5555-5555-5555-555555555555",
            },
        )
        favorite_id = res.get_json()["favorite"]["id"]

        res = client_b.get("/v1/me/favorites")
        self.assertEqual(res.get_json()["favorites"], [])
        res = client_b.delete(f"/v1/me/favorites/{favorite_id}")
        self.assertEqual(res.status_code, 404)

    def test_tag_create_list_and_assign(self):
        client = self._client_with_bearer()

        res = client.get("/v1/me/tags")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["tags"], [])
        self.assertIn("blue", body["available_colors"])

        res = client.post("/v1/me/tags", json={"name": "Important", "color": "red"})
        self.assertEqual(res.status_code, 201)
        red_tag = res.get_json()["tag"]
        self.assertEqual(red_tag["name"], "Important")

        res = client.post("/v1/me/tags", json={"name": "Watch", "color": "amber"})
        amber_tag = res.get_json()["tag"]

        res = client.post("/v1/me/tags", json={"name": "X", "color": "puce"})
        self.assertEqual(res.status_code, 400)

        res = client.post("/v1/me/tags", json={"name": "Important", "color": "red"})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.get_json()["created"])

        res = client.post(
            "/v1/me/favorites",
            json={
                "item_type": "agreement",
                "item_uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            },
        )
        favorite_id = res.get_json()["favorite"]["id"]

        res = client.put(
            f"/v1/me/favorites/{favorite_id}/tags",
            json={"tag_ids": [red_tag["id"], amber_tag["id"]]},
        )
        self.assertEqual(res.status_code, 200)
        names = sorted(t["name"] for t in res.get_json()["tags"])
        self.assertEqual(names, ["Important", "Watch"])

        # List favorites includes the tags
        favs = client.get("/v1/me/favorites").get_json()["favorites"]
        self.assertEqual(len(favs), 1)
        self.assertEqual(
            sorted(t["name"] for t in favs[0]["tags"]),
            ["Important", "Watch"],
        )

        # Filter favorites by tag_ids
        favs = client.get(
            f"/v1/me/favorites?tag_ids={red_tag['id']}"
        ).get_json()["favorites"]
        self.assertEqual(len(favs), 1)

        # tag that nothing matches
        res = client.post("/v1/me/tags", json={"name": "Other", "color": "blue"})
        other_id = res.get_json()["tag"]["id"]
        favs = client.get(
            f"/v1/me/favorites?tag_ids={other_id}"
        ).get_json()["favorites"]
        self.assertEqual(favs, [])

        # Replace assignments with a single tag
        res = client.put(
            f"/v1/me/favorites/{favorite_id}/tags",
            json={"tag_ids": [red_tag["id"]]},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.get_json()["tags"]), 1)

        # Delete a tag also removes it from the favorite
        res = client.delete(f"/v1/me/tags/{red_tag['id']}")
        self.assertEqual(res.status_code, 200)
        favs = client.get("/v1/me/favorites").get_json()["favorites"]
        self.assertEqual(favs[0]["tags"], [])

        # Delete the favorite cleans up assignments — re-assign first
        res = client.put(
            f"/v1/me/favorites/{favorite_id}/tags",
            json={"tag_ids": [amber_tag["id"]]},
        )
        self.assertEqual(res.status_code, 200)
        res = client.delete(f"/v1/me/favorites/{favorite_id}")
        self.assertEqual(res.status_code, 200)
        # Tag still exists, no orphan rows
        with self.app.app_context():
            engine = db.engines["auth"]
            with engine.begin() as conn:
                row = conn.execute(
                    text("SELECT COUNT(*) FROM favorite_tag_assignments")
                ).scalar()
                self.assertEqual(row, 0)

    def test_tag_user_isolation(self):
        client_a = self._client_with_bearer()
        token_b = self._create_user_and_bearer(email="b@example.com")
        client_b = self.app.test_client()
        client_b.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token_b}"

        res = client_a.post("/v1/me/tags", json={"name": "Mine", "color": "red"})
        a_tag_id = res.get_json()["tag"]["id"]

        # B can't update or delete A's tag
        res = client_b.patch(f"/v1/me/tags/{a_tag_id}", json={"color": "blue"})
        self.assertEqual(res.status_code, 404)
        res = client_b.delete(f"/v1/me/tags/{a_tag_id}")
        self.assertEqual(res.status_code, 404)

        # B can't assign A's tag to a B-owned favorite
        res = client_b.post(
            "/v1/me/favorites",
            json={
                "item_type": "agreement",
                "item_uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            },
        )
        b_fav_id = res.get_json()["favorite"]["id"]
        res = client_b.put(
            f"/v1/me/favorites/{b_fav_id}/tags",
            json={"tag_ids": [a_tag_id]},
        )
        self.assertEqual(res.status_code, 404)

    def test_bulk_add_and_remove_tags(self):
        client = self._client_with_bearer()

        res = client.post("/v1/me/tags", json={"name": "Review", "color": "blue"})
        review_tag = res.get_json()["tag"]
        res = client.post("/v1/me/tags", json={"name": "Urgent", "color": "red"})
        urgent_tag = res.get_json()["tag"]

        fav_ids: list[str] = []
        for idx in range(2):
            res = client.post(
                "/v1/me/favorites",
                json={
                    "item_type": "agreement",
                    "item_uuid": f"dddddddd-dddd-dddd-dddd-ddddddddddd{idx}",
                },
            )
            self.assertEqual(res.status_code, 201)
            fav_ids.append(res.get_json()["favorite"]["id"])

        res = client.post(
            "/v1/me/favorites/bulk-tags",
            json={
                "favorite_ids": fav_ids,
                "tag_ids": [review_tag["id"], urgent_tag["id"]],
                "action": "add",
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["updated"], 2)
        for favorite_id in fav_ids:
            self.assertEqual(
                sorted(tag["name"] for tag in body["tags_by_favorite"][favorite_id]),
                ["Review", "Urgent"],
            )

        res = client.post(
            "/v1/me/favorites/bulk-tags",
            json={
                "favorite_ids": fav_ids,
                "tag_ids": [urgent_tag["id"]],
                "action": "remove",
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        for favorite_id in fav_ids:
            self.assertEqual(
                [tag["name"] for tag in body["tags_by_favorite"][favorite_id]],
                ["Review"],
            )

    def test_bulk_tags_user_isolation(self):
        client_a = self._client_with_bearer()
        token_b = self._create_user_and_bearer(email="b@example.com")
        client_b = self.app.test_client()
        client_b.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token_b}"

        res = client_a.post("/v1/me/tags", json={"name": "Mine", "color": "blue"})
        a_tag_id = res.get_json()["tag"]["id"]
        res = client_b.post(
            "/v1/me/favorites",
            json={
                "item_type": "agreement",
                "item_uuid": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
            },
        )
        b_fav_id = res.get_json()["favorite"]["id"]

        res = client_b.post(
            "/v1/me/favorites/bulk-tags",
            json={
                "favorite_ids": [b_fav_id],
                "tag_ids": [a_tag_id],
                "action": "add",
            },
        )
        self.assertEqual(res.status_code, 404)

    def test_project_crud_filter_and_bulk_move(self):
        client = self._client_with_bearer()

        res = client.get("/v1/me/favorite-projects")
        self.assertEqual(res.status_code, 200)
        scratchpad = res.get_json()["projects"][0]
        self.assertEqual(scratchpad["color"], "slate")

        res = client.post(
            "/v1/me/favorite-projects",
            json={"name": "Diligence", "color": "teal"},
        )
        self.assertEqual(res.status_code, 201)
        diligence = res.get_json()["project"]
        self.assertEqual(diligence["name"], "Diligence")
        self.assertEqual(diligence["color"], "teal")

        res = client.patch(
            f"/v1/me/favorite-projects/{diligence['id']}",
            json={"name": "Board review", "color": "violet", "sort_order": 7},
        )
        self.assertEqual(res.status_code, 200)
        updated_project = res.get_json()["project"]
        self.assertEqual(updated_project["name"], "Board review")
        self.assertEqual(updated_project["color"], "violet")
        self.assertEqual(updated_project["sort_order"], 7)

        fav_ids: list[str] = []
        for idx in range(2):
            res = client.post(
                "/v1/me/favorites",
                json={
                    "item_type": "agreement",
                    "item_uuid": f"cccccccc-cccc-cccc-cccc-ccccccccccc{idx}",
                },
            )
            self.assertEqual(res.status_code, 201)
            fav_ids.append(res.get_json()["favorite"]["id"])

        res = client.post(
            "/v1/me/favorites/bulk-move",
            json={"favorite_ids": fav_ids, "project_id": diligence["id"]},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["moved"], 2)

        favs = client.get(
            f"/v1/me/favorites?project_id={diligence['id']}"
        ).get_json()["favorites"]
        self.assertEqual({fav["id"] for fav in favs}, set(fav_ids))
        self.assertEqual(set(favs[0]["project_ids"]), {diligence["id"]})

        res = client.post(
            "/v1/me/favorites/bulk-copy",
            json={"favorite_ids": [fav_ids[0]], "project_ids": [scratchpad["id"]]},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["copied"], 1)

        favs = client.get(
            f"/v1/me/favorites?project_id={scratchpad['id']}"
        ).get_json()["favorites"]
        self.assertEqual([fav["id"] for fav in favs], [fav_ids[0]])

        res = client.put(
            f"/v1/me/favorites/{fav_ids[0]}/projects",
            json={"project_ids": [scratchpad["id"]]},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["project_ids"], [scratchpad["id"]])

        res = client.delete(
            f"/v1/me/favorite-projects/{diligence['id']}"
            f"?reassign_project_id={scratchpad['id']}"
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["moved"], 1)

        favs = client.get(
            f"/v1/me/favorites?project_id={scratchpad['id']}"
        ).get_json()["favorites"]
        self.assertEqual({fav["id"] for fav in favs}, set(fav_ids))

    def test_project_user_isolation(self):
        client_a = self._client_with_bearer()
        token_b = self._create_user_and_bearer(email="b@example.com")
        client_b = self.app.test_client()
        client_b.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token_b}"

        res = client_a.post(
            "/v1/me/favorite-projects",
            json={"name": "Mine", "color": "green"},
        )
        project_id = res.get_json()["project"]["id"]

        res = client_b.patch(
            f"/v1/me/favorite-projects/{project_id}",
            json={"name": "Not mine"},
        )
        self.assertEqual(res.status_code, 404)
        res = client_b.delete(f"/v1/me/favorite-projects/{project_id}")
        self.assertEqual(res.status_code, 404)

    def test_invalid_item_type_rejected(self):
        client = self._client_with_bearer()
        res = client.post(
            "/v1/me/favorites",
            json={"item_type": "garbage", "item_uuid": "x"},
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
