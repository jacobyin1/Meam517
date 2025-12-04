import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import jax.numpy as jnp


def save_optimized_xml(xml_path, output_path, model, included_ids, end_ids, shifts):
    """
    Patches the original XML with the optimized body positions.

    Args:
        xml_path: Path to the original (source) XML.
        output_path: Where to save the new XML.
        model: The ORIGINAL mujoco.MjModel (needed for names).
        included_ids: List of body IDs that were optimized (intermediate bodies).
        end_ids: List of body IDs that were optimized (end effectors).
        shifts: The JAX array of optimal shifts [included_shifts..., end_shifts...].
    """
    print(f"Reading XML from: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Convert JAX array to Numpy for easier handling
    if hasattr(shifts, 'device'):
        shifts = np.array(shifts)

    # 1. Update Intermediate Bodies
    # These match the first part of the 'shifts' array
    for i, body_id in enumerate(included_ids):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        shift = shifts[i]

        _update_xml_body_pos(root, body_name, shift)

    # 2. Update End Effectors
    # These match the second part of the 'shifts' array
    offset = len(included_ids)
    for i, body_id in enumerate(end_ids):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        shift = shifts[offset + i]

        _update_xml_body_pos(root, body_name, shift)

    print(f"Saving optimized robot to: {output_path}")
    tree.write(output_path)


def _update_xml_body_pos(root, body_name, shift):
    """Helper to find a body tag and add the shift to its 'pos'."""
    # Find the body tag with specific name
    # XPath: .//body[@name='body_name']
    found = False
    for body in root.findall(f".//body[@name='{body_name}']"):
        found = True

        # Parse current pos
        # pos="0 0 -0.5" -> [0.0, 0.0, -0.5]
        current_pos_str = body.get('pos')
        if current_pos_str is None:
            # Default to 0 0 0 if missing
            current_pos = np.array([0.0, 0.0, 0.0])
        else:
            current_pos = np.array([float(x) for x in current_pos_str.split()])

        # Add Shift
        new_pos = current_pos + shift

        # Format back to string
        # pos="0.0 0.0 -0.8"
        new_pos_str = f"{new_pos[0]:.4f} {new_pos[1]:.4f} {new_pos[2]:.4f}"
        body.set('pos', new_pos_str)

        # Optional: Print for debugging
        # print(f"Updated {body_name}: {current_pos_str} -> {new_pos_str}")

    if not found:
        print(f"WARNING: Optimized body '{body_name}' not found in XML structure.")