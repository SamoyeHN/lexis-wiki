### SYSTEM ###
You are an expert Educational Content Designer and Mind Mapping Specialist.
### USER ###
Construct a highly detailed hierarchical mind map representing the structural ideas, themes, and supporting details of the text.

MANDATES:
1. Establish a single central root theme summarizing the unit and output it in `root_name`.
2. Extract exactly 3 to 5 distinct primary branches representing the major sub-themes or narrative stages. Assign each a unique, harmonious color theme from the allowed values.
3. Consistent Hierarchical Structure: Every branch MUST contain one or more sub-branch objects. Each sub-branch has a clear `sub_branch_name` and a `leaves` array. If a branch covers only a single general topic, simply name its sub-branch 'Overview' or 'Key Points'.
4. Concise, High-Impact Leaves: Leaf nodes should be concise bullet-point details, phrases, or short examples (aim for 3-10 words per leaf). DO NOT copy long, multi-line paragraphs as leaf nodes.

CONTENT:
{content}
