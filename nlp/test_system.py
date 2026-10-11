"""
Comprehensive Test Suite for SentinelAI Toxicity Detector.
Tests model inference, FastAPI endpoints in main.py, and api.py.
"""
import sys
import unittest
from fastapi.testclient import TestClient

import main
import api

class TestSentinelAISystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_client = TestClient(main.app)
        cls.api_client = TestClient(api.app)

    def test_main_health(self):
        resp = self.main_client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "online")
        self.assertTrue(data.get("model_loaded"))

    def test_api_health(self):
        resp = self.api_client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "online")
        self.assertTrue(data.get("model_loaded"))

    def test_main_safe_comment(self):
        text = "I completely disagree with your point, but let's discuss it in class tomorrow so I can understand your perspective."
        resp = self.main_client.post("/analyze", json={"text": text})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_toxic"], f"Expected safe, got toxic: {data}")
        self.assertLess(data["confidence_score"], 40.0)
        self.assertIn("approved", data["action"].lower())

    def test_main_toxic_insult(self):
        text = "You are an absolute idiot and a loser, shut up!"
        resp = self.main_client.post("/analyze", json={"text": text})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_toxic"], f"Expected toxic, got safe: {data}")
        self.assertGreaterEqual(data["confidence_score"], 50.0)
        self.assertIn("blocked", data["action"].lower())

    def test_main_physical_threat(self):
        text = "I will hurt you and beat you up if you come to class."
        resp = self.main_client.post("/analyze", json={"text": text})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_toxic"], f"Expected toxic, got safe: {data}")
        self.assertGreaterEqual(data["confidence_score"], 80.0)

    def test_api_safe_and_toxic(self):
        safe_resp = self.api_client.post("/analyze", json={"text": "Thank you for explaining the math problem, teacher!"})
        self.assertEqual(safe_resp.status_code, 200)
        self.assertFalse(safe_resp.json()["is_toxic"])

        toxic_resp = self.api_client.post("/analyze", json={"text": "You are a stupid idiot, shut up and get lost!"})
        self.assertEqual(toxic_resp.status_code, 200)
        self.assertTrue(toxic_resp.json()["is_toxic"])

    def test_empty_input_validation(self):
        resp = self.main_client.post("/analyze", json={"text": "   "})
        self.assertEqual(resp.status_code, 400)

        resp2 = self.api_client.post("/analyze", json={"text": ""})
        self.assertEqual(resp2.status_code, 400)

    def test_serve_html(self):
        resp = self.main_client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("SentinelAI", resp.text)

        resp2 = self.api_client.get("/")
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("SentinelAI", resp2.text)

if __name__ == "__main__":
    unittest.main()

