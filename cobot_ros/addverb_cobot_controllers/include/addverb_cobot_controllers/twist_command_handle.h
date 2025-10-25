
#ifndef ADDVERB_TWIST_COMMAND_HANDLE_H
#define ADDVERB_TWIST_COMMAND_HANDLE_H

#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
#include <hardware_interface/internal/hardware_resource_manager.h>

namespace addverb
{
class TwistCommandHandle
{
public:
    TwistCommandHandle() = default;
    TwistCommandHandle(std::string& frame_id,geometry_msgs::Twist* cmd): name_(frame_id), cmd_(cmd)
    {
        if(!cmd)
        {
          ROS_ERROR("command pointer was null can not create handle");
        }
    }
    
    ~TwistCommandHandle() = default;

    std::string getName() const 
    {
        return name_;
    }

    void setCommand(const geometry_msgs::Twist& twist)
    {
      assert(cmd_);
      *cmd_ = twist;
    }

    geometry_msgs::Twist getCommand() const
    {
      assert(cmd_);
      return *cmd_;
    }

    const geometry_msgs::Twist* getCommandPtr() const
    {
      assert(cmd_);
      return cmd_;
    }

private:
    geometry_msgs::Twist* cmd_ = { nullptr };
    std::string name_;

};

class TwistCommandInterface
  : public hardware_interface::HardwareResourceManager<TwistCommandHandle, hardware_interface::ClaimResources>
{
};
} // namespace addverb

#endif
