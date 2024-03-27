#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>

typedef struct
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
} Bbox;

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

std::vector<int> TopKIndex(const std::vector<float> &vec, int topk)
{
    std::vector<int> topKIndex;
    topKIndex.clear();

    std::vector<size_t> vec_index(vec.size());
    std::iota(vec_index.begin(), vec_index.end(), 0);

    std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2)
              { return vec[index_1] > vec[index_2]; });

    int k_num = std::min<int>(vec.size(), topk);

    for (int i = 0; i < k_num; ++i)
    {
        topKIndex.emplace_back(vec_index[i]);
    }

    return topKIndex;
}

int sub2ind(const int row, const int col, const int cols, const int rows)
{
    return row * cols + col;
}

void ind2sub(const int sub, const int cols, const int rows, int &row, int &col)
{
    row = sub / cols;
    col = sub % cols;
}

float GetIoU(const Bbox box1, const Bbox box2)
{
    int x1 = std::max(box1.xmin, box2.xmin);
    int y1 = std::max(box1.ymin, box2.ymin);
    int x2 = std::min(box1.xmax, box2.xmax);
    int y2 = std::min(box1.ymax, box2.ymax);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

std::vector<int> multiclass_nms_class_agnostic(std::vector<Bbox> boxes, std::vector<float> confidences, const float nms_thresh)
{
    std::sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2)
              { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    std::vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    std::vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}

bool isZero(int num) { return num == 0; }
std::vector<int> multiclass_nms_class_aware(const std::vector<Bbox> boxes, const std::vector<float> confidences, const std::vector<int> labels, const float nms_thresh, const int num_classes)
{
    const int num_box = boxes.size();
    std::vector<int> keep(num_box, 0);
    for (int i = 0; i < num_classes; i++)
    {
        std::vector<int> inds;
        std::vector<Bbox> c_bboxes;
        std::vector<float> c_scores;
        for (int j = 0; j < labels.size(); j++)
        {
            if (labels[j] == i)
            {
                inds.emplace_back(j);
                c_bboxes.emplace_back(boxes[j]);
                c_scores.emplace_back(confidences[j]);
            }
        }
        if (inds.size() == 0)
        {
            continue;
        }
        
        std::vector<int> c_keep = multiclass_nms_class_agnostic(c_bboxes, c_scores, nms_thresh);
        for (int j = 0; j < c_keep.size(); j++)
        {
            keep[inds[c_keep[j]]] = 1;
        }
    }
    
    std::vector<int> keep_inds;
    for (int i = 0; i < keep.size(); i++)
    {
        if (keep[i] > 0)
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}


static const char *ucf24_labels[] = {"Basketball", "BasketballDunk", "Biking", "CliffDiving", "CricketBowling", "Diving", "Fencing", "FloorGymnastics", "GolfSwing", "HorseRiding", "IceDancing", "LongJump", "PoleVault", "RopeClimbing", "SalsaSpin", "SkateBoarding", "Skiing", "Skijet", "SoccerJuggling", "Surfing", "TennisSwing", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog"};
static const char *ava_labels[] = {"bend/bow(at the waist)", "crawl", "crouch/kneel", "dance", "fall down", "get up", "jump/leap", "lie/sleep", "martial art", "run/jog", "sit", "stand", "swim", "walk", "answer phone", "brush teeth", "carry/hold (an object)", "catch (an object)", "chop", "climb (e.g. a mountain)", "clink glass", "close (e.g., a door, a box)", "cook", "cut", "dig", "dress/put on clothing", "drink", "drive (e.g., a car, a truck)", "eat", "enter", "exit", "extract", "fishing", "hit (an object)", "kick (an object)", "lift/pick up", "listen (e.g., to music)", "open (e.g., a window, a car door)", "paint", "play board game", "play musical instrument", "play with pets", "point to (an object)", "press", "pull (an object)", "push (an object)", "put down", "read", "ride (e.g., a bike, a car, a horse)", "row boat", "sail boat", "shoot", "shovel", "smoke", "stir", "take a photo", "text on/look at a cellphone", "throw", "touch (an object)", "turn (e.g., a screwdriver)", "watch (e.g., TV)", "work on a computer", "write", "fight/hit (a person)", "give/serve (an object) to (a person)", "grab (a person)", "hand clap", "hand shake", "hand wave", "hug (a person)", "kick (a person)", "kiss (a person)", "lift (a person)", "listen to (a person)", "play with kids", "push (another person)", "sing to (e.g., self, a person, a group)", "take (an object) from (a person)", "talk to (e.g., self, a person, a group)", "watch (a person)"};