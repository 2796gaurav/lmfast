import unittest

from pydantic import BaseModel

from lmfast.utils.structured import generate_structured


class User(BaseModel):
    name: str
    age: int


class TestStructured(unittest.TestCase):
    def test_structured_gen(self):
        # Mock LLM returning JSON code block
        def mock_gen(prompt):
            return 'Here is the JSON:\n```json\n{"name": "Alice", "age": 30}\n```'

        user = generate_structured(mock_gen, "Make user", User)

        self.assertIsNotNone(user)
        self.assertEqual(user.name, "Alice")
        self.assertEqual(user.age, 30)


if __name__ == "__main__":
    unittest.main()
