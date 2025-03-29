def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)


def union_ranges(ranges):
    # Sort ranges based on the starting index
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    # Initialize with the first range
    merged_ranges = [sorted_ranges[0]]

    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]

        # Check if the current range overlaps or is contiguous with the last range in the merged list
        if current_start <= last_end:
            # Merge the two ranges
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add the current range as new
            merged_ranges.append((current_start, current_end))

    return merged_ranges


def intersect_two_ranges(range1, range2):
    # Unpack the ranges
    start1, end1 = range1
    start2, end2 = range2

    # Calculate the maximum of the starting indices and the minimum of the ending indices
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)

    # Check if the intersection is valid (the start is less than or equal to the end)
    if intersect_start <= intersect_end:
        return intersect_start, intersect_end
    else:
        return None  # Return an None if there is no intersection




class Evaluator:
    def __init__(self, questions_df):
        self.questions_df = questions_df


    def precision_recall_scores(self, question_metadatas):
        precision_scores = []
        recall_scores = []

        for (index, row), metadatas in zip(self.questions_df.iterrows(), question_metadatas):
            references = row['references']

            numerator_sets = []

            for metadata in metadatas:
                chunk_start, chunk_end, = metadata['start_idx'], metadata['end_idx']

                for ref_obj in references:
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])

                    intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))

                    if intersection is not None:
                        numerator_sets = union_ranges([intersection] + numerator_sets)

            if numerator_sets:
                numerator_value = sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in references])
            precision_denominator = sum_of_ranges([(x['start_idx'], x['end_idx']) for x in metadatas])

            recall_score = numerator_value / recall_denominator
            recall_scores.append(recall_score)

            precision_score = numerator_value / precision_denominator
            precision_scores.append(precision_score)

        return precision_scores, recall_scores
