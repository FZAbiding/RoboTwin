"""Read-only diagnostics; none of these values are learner inputs."""
import numpy as np
from envs.utils import get_face_prod

def plain(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if hasattr(value,'detach'):return value.detach().cpu().tolist()
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [plain(v) for v in value]
 if value is None or isinstance(value,(str,int,float,bool)):return value
 if hasattr(value,'p') and hasattr(value,'q'):return {'p':plain(value.p),'q':plain(value.q)}
 return str(value)

class Observer:
 def __init__(self,event):self.event=event;self.phase=None;self.seed=None;self.stage=0;self.action=0
 def emit(self,kind,**fields):self.event(kind,seed=self.seed,phase=self.phase,stage=self.stage,**plain(fields))
 def metrics(self,environment):
  pot=environment.pot;pose=pot.get_pose();contacts=[np.asarray(pot.get_contact_point(i)[:3]) for i in (0,1)];tcps=[np.asarray(getattr(environment.robot,f'get_{arm}_tcp_pose')()[:3]) for arm in ('left','right')];dist=[float(np.linalg.norm(t-c)) for t,c in zip(tcps,contacts)];alignment=float(get_face_prod(pose.q,[0,0,1],[0,0,1]));height=float(pose.p[2]);conditions=[height>.82,dist[0]<.03,dist[1]<.03,alignment>.8]
  result={'pot_pose':plain(pose),'contact_world_xyz':[v.tolist() for v in contacts],'tcp_world_xyz':[v.tolist() for v in tcps],'pot_height_m':height,'tcp_contact_distance_m':dist,'orientation_dot':alignment,'original_predicate_components':conditions,'original_predicate_conjunction':bool(all(conditions)),'gripper_commands':[float(environment.robot.get_left_gripper_val()),float(environment.robot.get_right_gripper_val())]}
  pot_names={link.entity.name for link in pot.actor.get_links()};physical=[]
  for contact in environment.scene.get_contacts():
   names=[body.entity.name for body in contact.bodies]
   if not any(n in pot_names for n in names):continue
   impulses=[]
   for point in contact.points:
    if hasattr(point,'impulse'):impulses.append(float(np.linalg.norm(point.impulse)))
   physical.append({'bodies':names,'points':len(contact.points),'sum_impulse_norm':sum(impulses) if impulses else None})
  result['physical_contacts_with_pot_links']=physical;return result
 def wrap(self,owner,name,label,describe=None):
  if not hasattr(owner,name):return
  method=getattr(owner,name);original=getattr(method,'_robotwin_diagnostic_original',method)
  def observed(*args,**kwargs):
   context=self.phase=='expert'
   if context and label=='expert_move':self.stage+=1
   if context:self.emit(label+'_begin',arguments=describe(args,kwargs) if describe else {},method=name)
   try:value=original(*args,**kwargs)
   except BaseException as error:
    if context:self.emit(label+'_error',method=name,error=f'{type(error).__name__}: {error}')
    raise
   if context:
    result={}
    if isinstance(value,dict):result={key:value[key] for key in ('status','position','success') if key in value and key!='position'}
    elif label=='curobo':result={key:getattr(value,key,None) for key in ('status','success','valid_query','position_error','rotation_error','ik_time','trajopt_time','total_time')}
    elif label=='grasp_choice':result={'chosen_pose':value}
    if label=='expert_move':result['plan_success']=bool(self.environment.plan_success);result['metrics']=self.metrics(self.environment)
    self.emit(label+'_end',method=name,result=result)
   return value
  observed._robotwin_diagnostic_original=original;setattr(owner,name,observed)
 def on_setup(self,environment,phase,seed):
  self.environment=environment;self.phase=phase;self.seed=seed;self.stage=0;self.action=0
  self.emit('diagnostic_initial_state',model_id=int(environment.model_id),model_name=environment.model_name,metrics=self.metrics(environment))
  def describe_actions(args,kwargs):
   groups=[]
   for group in args[:2]:
    if group is None:continue
    arm,actions=group;groups.append({'arm':str(arm),'actions':[{'type':v.action,'target_pose':v.target_pose,'gripper_command':v.target_gripper_pos} for v in actions if v is not None]})
   return {'groups':groups}
  self.wrap(environment,'move','expert_move',describe_actions)
  self.wrap(environment,'choose_best_pose','grasp_choice',lambda a,k:{'raw_grasp_pose':a[0] if a else None,'center_pose':a[1] if len(a)>1 else None,'arm':str(a[2]) if len(a)>2 else str(k.get('arm_tag'))})
  for arm in ('left','right'):
   for method in (f'{arm}_plan_path',f'{arm}_plan_multi_path'):
    self.wrap(environment.robot,method,'robot_plan',lambda a,k:{'target':a[0] if a else None,'constraint_pose':k.get('constraint_pose')})
   planner=getattr(environment.robot,f'{arm}_planner',None)
   if planner:
    for attr,method in [('motion_gen','plan_single'),('motion_gen_batch','plan_batch')]:
     instance=getattr(planner,attr,None)
     if instance:self.wrap(instance,method,'curobo')
 def on_expert_end(self,environment):self.emit('diagnostic_expert_end',plan_success=bool(environment.plan_success),metrics=self.metrics(environment))
 def on_policy_action(self,environment):
  self.action+=1;self.emit('diagnostic_policy_state',action_index=self.action,metrics=self.metrics(environment))
 def on_trial_end(self,environment,result):self.emit('diagnostic_trial_end',trial=result,metrics=self.metrics(environment))
