"""Test offline del client Solem: estrattori HTML, scaling e logica di login (nessuna rete)."""
import os, sys, unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import water.Ingestion.solem_client as sc

PAGE = '''<script>const userId = "0123456789abcdef01234567";</script>
<script>let module = {"id":"aaaaaaaaaaaaaaaaaaaaaaaa","name":"Donna Elvira {x}","outputs":[{"id":"o1"}]};
var inputs = [{"id":"bbbbbbbbbbbbbbbbbbbbbbbb","type":1,"name":"","expression":"x/0.1"},
              {"id":"cccccccccccccccccccccccc","type":2,"name":"T","expression":"x/100.0"}];</script>
<input type="text" name="sensor-name" value="Flowmeter" data-input-id="bbbbbbbbbbbbbbbbbbbbbbbb" />'''


class FakeResp:
    def __init__(self, url="https://mysolem.com/x", text="", status=200, js=None):
        self.url, self.text, self.status_code, self._js = url, text, status, js
    def json(self):
        if self._js is None: raise ValueError("no json")
        return self._js
    def raise_for_status(self): pass


class FakeSession:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def post(self, url, **kw): self.calls.append(("POST", url)); return self.responses.pop(0)
    def get(self, url, **kw): self.calls.append(("GET", url)); return self.responses.pop(0)
    def request(self, method, url, **kw): self.calls.append((method, url)); return self.responses.pop(0)


def make_client(responses):
    c = sc.SolemClient(email="a@b.it", password="x", pause_s=0)
    c.session = FakeSession(responses)
    return c


class Parsing(unittest.TestCase):
    def test_userid(self):
        self.assertEqual(sc.SolemClient._parse_user_id(PAGE), "0123456789abcdef01234567")

    def test_module_and_inputs(self):
        c = make_client([FakeResp(text=PAGE)]); c.user_id = "u"
        module, inputs = c.get_module_page("m")
        self.assertEqual(module["id"], "a" * 24)
        self.assertEqual(module["name"], "Donna Elvira {x}")          # parentesi dentro le stringhe
        self.assertEqual(inputs[0]["name"], "Flowmeter")             # etichetta dal form
        self.assertEqual([i["id"] for i in sc.SolemClient.flow_inputs(inputs)], ["b" * 24])

    def test_expression(self):
        self.assertEqual(sc.apply_expression("x/0.1", 5), 50.0)
        self.assertEqual(sc.apply_expression("x*2", 5), 10.0)
        self.assertEqual(sc.apply_expression(None, 5), 5.0)
        self.assertIsNone(sc.apply_expression("__import__('os')", 5))
        self.assertIsNone(sc.apply_expression("x/0", 5))

    def test_js_iso(self):
        self.assertEqual(sc._js_iso(datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)), "2026-07-01T12:00:00.000Z")
        with self.assertRaises(ValueError):
            sc._js_iso(datetime(2026, 7, 1))


class Auth(unittest.TestCase):
    def test_login_ok_and_bad_credentials(self):
        c = make_client([FakeResp(text=PAGE)])
        self.assertEqual(c.login(), "0123456789abcdef01234567")
        bad = make_client([FakeResp(text="no id"), FakeResp(text="no id")])
        with self.assertRaises(sc.SolemAuthError):
            bad.login()

    def test_relogin_once_on_expired_session(self):
        c = make_client([
            FakeResp(url="https://mysolem.com/login"),           # sessione scaduta
            FakeResp(text=PAGE),                                 # nuovo login
            FakeResp(js={"tick": {"value": 3}}),                 # ripetizione della richiesta
        ])
        c.user_id = "old"
        self.assertEqual(c.get_last_tick("b" * 24), {"value": 3})

    def test_no_infinite_relogin(self):
        c = make_client([FakeResp(url="https://mysolem.com/login"), FakeResp(text=PAGE),
                         FakeResp(url="https://mysolem.com/login")])
        c.user_id = "old"
        with self.assertRaises(sc.SolemAuthError):
            c.get_last_tick("b" * 24)

    def test_missing_credentials(self):
        # Il costruttore chiama load_dotenv(): senza questo patch un .env reale rimetterebbe
        # le credenziali nell'ambiente e il test passerebbe o fallirebbe a seconda della macchina.
        with mock.patch.object(sc, "load_dotenv"), mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                sc.SolemClient(email=None, password=None)


if __name__ == "__main__":
    unittest.main()