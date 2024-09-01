from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # read video
    video_frames = read_video("input_videos/football-input.mp4")

    # Initialize Tracker
    tracker = Tracker('/Users/zoghby.marc/Desktop/pv/weights/dataset-v3/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['color'] = team_assigner.team_colors[team]
    
    # Draw Output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save Video
    save_video(output_video_frames, "output_videos/football-output-updated.avi")
    print('Done!')


if __name__ == "__main__":
    main()