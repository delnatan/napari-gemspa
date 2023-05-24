from qtpy.QtWidgets import QMessageBox


def remove_outside_mask(df, labeled_mask, id_column='index'):

    if id_column == 'index' and 'index' not in df.columns:
        df['index'] = df.index
    if id_column not in df.columns:
        raise ValueError(f"Error: column '{id_column}' not found in data frame.")

    drop_idx = []
    if len(labeled_mask.shape) == 2:
        for row in df.iterrows():
            if not labeled_mask[int(row[1]['y'])][int(row[1]['x'])]:
                drop_idx.append(row[1][id_column])
    elif len(labeled_mask.shape) == 3:
        # assume first dimension is frame
        for row in df.iterrows():
            if not labeled_mask[int(row[1]['frame'])][int(row[1]['y'])][int(row[1]['x'])]:
                drop_idx.append(row[1][id_column])
    else:
        raise ValueError(f"Cannot use mask: dimension is {len(labeled_mask.shape)} (expecting 2 or 3).")

    df = df[~df[id_column].isin(drop_idx)]
    df.index = range(len(df))
    return df


def convert_to_float(value):
    if value:
        return float(value)
    else:
        return None


def convert_to_int(value):
    if value:
        return int(value)
    else:
        return None


def fix_frame_limits(frame_start, frame_end, num_frames):
    if frame_start is None or frame_start < 0 or frame_start >= num_frames:
        frame_start = 0
    if frame_end is None or frame_end < frame_start:
        frame_end = num_frames - 1
    return frame_start, frame_end


def show_error(message):
    """Display an error message in a QMessage box

    Parameters
    ----------
    message: str
        Error message

    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)

    msg.setText(message)
    msg.setWindowTitle("GEMspa error")
    msg.exec_()