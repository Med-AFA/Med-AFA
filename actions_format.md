# actions.json Format

`actions.json` defines the information groups that can be acquired by the policy. Pass its path through `--actions_path`.

The JSON root may be either an array of actions or an object with an `actions` array. Use the array form below as the canonical format.

```json
[
  {
    "action_id": "action_1",
    "feature": ["feature_1", "feature_2"],
    "prerequisites": []
  },
  {
    "action_id": "action_2",
    "feature": ["feature_3"],
    "prerequisites": ["action_1"]
  }
]
```

Each action must include the following fields.

| Field             | Type             | Requirement                                                                                                   |
| ----------------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `action_id`     | string           | A unique, stable identifier. It is referenced by`prerequisites`; provide it explicitly for every action.    |
| `feature`       | array of strings | One or more CSV feature-column names revealed by this action.                                                 |
| `prerequisites` | array of strings | The`action_id` values that must be acquired before this action. Use `[]` when no predecessor is required. |

Requirements:

- Every entry in `feature` must exactly match a non-label CSV column.
- An action must contain at least one feature.
- A feature may belong to only one action. In the standard pipeline, every non-label CSV feature must be assigned to exactly one action.
- Every prerequisite must reference an existing `action_id`. An action cannot require itself, and prerequisite relations should form a directed acyclic graph.
- List order defines the stable action index used in saved outputs; keep it fixed across reruns of the same dataset.
- Do not include the label column in any action.

The prerequisite relation is enforced as a hard mask during action selection: an action is legal only after all listed prerequisite actions have been selected.
